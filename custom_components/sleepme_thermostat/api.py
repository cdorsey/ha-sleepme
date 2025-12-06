"""Sleep.me API client module."""

from __future__ import annotations

import logging
import socket
from decimal import Decimal
from http import HTTPMethod, HTTPStatus
from typing import Any, Literal

import aiohttp
import async_timeout
from pydantic import BaseModel, model_serializer, model_validator

from .const import LOGGER
from .rate_limiter import RateLimiter

TIMEOUT = 10


_LOGGER: logging.Logger = logging.getLogger(__package__)

HEADERS = {"Content-type": "application/json; charset=UTF-8"}


class SleepmeApiClientError(Exception):
    """Exception to indicate a general API error."""


class SleepmeApiClientCommunicationError(
    SleepmeApiClientError,
):
    """Exception to indicate a communication error."""


class SleepmeApiClientAuthenticationError(
    SleepmeApiClientError,
):
    """Exception to indicate an authentication error."""


class SleepmeApiClientRateLimitError(
    SleepmeApiClientError,
):
    """Exception to indicate a rate limit error."""


def _verify_response_or_raise(response: aiohttp.ClientResponse) -> None:
    """Verify that the response is valid."""
    if response.status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
        msg = "Invalid credentials"
        raise SleepmeApiClientAuthenticationError(
            msg,
        )
    if response.status == HTTPStatus.TOO_MANY_REQUESTS:
        msg = "Rate limit exceeded"
        raise SleepmeApiClientRateLimitError(
            msg,
        )
    response.raise_for_status()


class SleepmeDevice(BaseModel):
    """Data for a Sleep.me device."""

    id: str
    name: str
    attachments: list[str] = []

    @property
    def is_chilipad_pro(self) -> bool:
        return "CHILIPAD_PRO" in self.attachments


class SleepmeDevices(BaseModel):
    devices: list[SleepmeDevice]

    @model_validator(mode="before")
    @classmethod
    def validate_devices(cls, data: Any) -> Any:
        if isinstance(data, list):
            return {"devices": data}

        return data

    @model_serializer
    def serialize_devices(self) -> Any:
        return self.devices


class SleepmeDeviceAbout(BaseModel):
    firmware_version: str
    ip_address: str
    lan_address: str
    mac_address: str
    model: str
    serial_number: str


class SleepmeDeviceControl(BaseModel):
    brightness_level: int
    display_temperature_unit: Literal["f", "c"]
    set_temperature_c: Decimal
    set_temperature_f: Decimal
    thermal_control_status: Literal["active", "standby"]
    time_zone: str


class SleepmeDeviceStatus(BaseModel):
    is_connected: bool
    is_water_low: bool
    water_level: int
    water_temperature_c: Decimal
    water_temperature_f: Decimal


class SleepmeDeviceState(BaseModel):
    """Data for a Sleep.me device state."""

    about: SleepmeDeviceAbout
    control: SleepmeDeviceControl
    status: SleepmeDeviceStatus


class SleepmeApiClient:
    """Sleep.me API client for interacting with the Sleep.me service."""

    def __init__(
        self,
        api_key: str,
        session: aiohttp.ClientSession,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        """Sleep.me API Client."""
        self._api_key = api_key
        self._session = session
        self._rate_limiter = rate_limiter or RateLimiter()

    async def async_get_data(self) -> list[dict]:
        """Get data from the API."""

        return [device.model_dump() for device in await self.async_get_devices()]

    async def async_get_devices(self) -> list[SleepmeDevice]:
        """Get devices from the API."""
        url = "https://api.developer.sleep.me/v1/devices"
        devices = await self.api_wrapper(HTTPMethod.GET, url, SleepmeDevices)

        return [device for device in devices.devices if device.is_chilipad_pro]

    async def async_get_device_state(self, device_id: str) -> SleepmeDeviceState:
        """Get device state from the API."""

        url = f"https://api.developer.sleep.me/v1/devices/{device_id}"
        return await self.api_wrapper(HTTPMethod.GET, url, SleepmeDeviceState)

    async def async_set_device_temperature(
        self, device_id: str, temperature: int
    ) -> SleepmeDeviceControl:
        """Set device temperature from the API."""

        url = f"https://api.developer.sleep.me/v1/devices/{device_id}"

        return await self.api_wrapper(
            HTTPMethod.PATCH,
            url,
            SleepmeDeviceControl,
            data={"set_temperature_f": temperature},
        )

    async def async_set_device_mode(
        self, device_id: str, mode: str
    ) -> SleepmeDeviceControl:
        """Set device mode from the API."""

        url = f"https://api.developer.sleep.me/v1/devices/{device_id}"
        return await self.api_wrapper(
            HTTPMethod.PATCH,
            url,
            SleepmeDeviceControl,
            data={"thermal_control_status": mode},
        )

    async def api_wrapper[T: BaseModel](
        self,
        method: HTTPMethod,
        url: str,
        response_model: type[T],
        *,
        data: dict | None = None,
    ) -> T:
        """Get information from the API."""
        if data is None:
            data = {}
        headers = HEADERS.copy()
        headers["Authorization"] = f"Bearer {self._api_key}"

        if not self._rate_limiter.can_send_request():
            LOGGER.info("Rate limit exceeded")

        try:
            async with async_timeout.timeout(10):
                response = await self._session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                )
                _verify_response_or_raise(response)

                content = await response.read()

                LOGGER.debug(f"{method} {url} response: {content}")

                return response_model.model_validate_json(content)

        except TimeoutError as exception:
            msg = f"Timeout error fetching information - {exception}"
            raise SleepmeApiClientCommunicationError(
                msg,
            ) from exception
        except (aiohttp.ClientError, socket.gaierror) as exception:
            msg = f"Error fetching information - {exception}"
            raise SleepmeApiClientCommunicationError(
                msg,
            ) from exception
        except Exception as exception:  # pylint: disable=broad-except
            msg = f"Something really wrong happened! - {exception}"
            raise SleepmeApiClientError(
                msg,
            ) from exception
