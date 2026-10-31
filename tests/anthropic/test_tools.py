from __future__ import annotations

import asyncio
from datetime import date
from enum import Enum
import json
from typing import Any

from pydantic import BaseModel
import pytest

from orkexa import Tool, ToolDefinitionError, ToolExecutionError


class Unit(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"


class Location(BaseModel):
    city: str
    country: str


class Forecast(BaseModel):
    temperature: float
    unit: Unit


@Tool
def get_forecast(location: Location, day: date, unit: Unit = Unit.CELSIUS) -> Forecast:
    """Get the weather forecast for a location.

    Args:
        location: Where to look up the weather.
        day (date): The day of the forecast.
        unit: Temperature unit of the
            returned forecast.

    Returns:
        The forecast.
    """
    assert isinstance(location, Location)
    assert isinstance(day, date)
    return Forecast(temperature=21.5, unit=unit)


def test_schema_is_derived_from_annotations_and_docstring() -> None:
    schema = get_forecast.schema()
    assert schema["name"] == "get_forecast"
    assert schema["strict"] is True
    assert schema["description"] == "Get the weather forecast for a location.\n\nReturns: The forecast."
    input_schema = schema["input_schema"]
    assert input_schema["additionalProperties"] is False
    assert input_schema["required"] == ["location", "day"]
    props = input_schema["properties"]
    assert props["unit"]["description"] == "Temperature unit of the returned forecast."
    assert props["day"]["format"] == "date"
    assert input_schema["$defs"]["Location"]["additionalProperties"] is False


def test_schema_returns_a_copy() -> None:
    get_forecast.schema()["input_schema"]["properties"].clear()
    assert get_forecast.schema()["input_schema"]["properties"]


def test_execute_accepts_json_shaped_input() -> None:
    # Enums, dates and nested models arrive as plain JSON values from the API.
    output = get_forecast.execute(
        {
            "location": {"city": "Wien", "country": "AT"},
            "day": "2026-10-27",
            "unit": "fahrenheit",
        }
    )
    assert json.loads(output) == {"temperature": 21.5, "unit": "fahrenheit"}


def test_direct_call_bypasses_validation() -> None:
    result = get_forecast(Location(city="Graz", country="AT"), date(2026, 1, 1))
    assert result.unit is Unit.CELSIUS
    assert get_forecast.__name__ == "get_forecast"  # type: ignore[attr-defined]


def test_string_return_is_passed_through() -> None:
    @Tool(name="echo-tool")
    def echo(text: str) -> str:
        """Echo text.

        Args:
            text: Text to echo.

        Returns:
            The same text.
        """
        return text

    assert echo.name == "echo-tool"
    assert echo.execute({"text": "hallo"}) == "hallo"


@pytest.mark.parametrize(
    "payload",
    [{}, {"a": "1"}, {"a": 1, "extra": True}, {"a": 1.5}],
)
def test_invalid_input_raises(payload: dict[str, Any]) -> None:
    @Tool()
    def square(a: int) -> int:
        """Square a number.

        Args:
            a: The number.

        Returns:
            a squared.
        """
        return a * a

    with pytest.raises(ToolExecutionError, match="invalid tool inputs"):
        square.execute(payload)


def test_handler_failure_and_invalid_return_raise() -> None:
    @Tool
    def broken(a: int) -> int:
        """Always fails.

        Args:
            a: Ignored.

        Returns:
            Nothing.
        """
        raise ValueError("boom")

    @Tool
    def wrong_type(a: int) -> int:
        """Returns the wrong type.

        Args:
            a: Ignored.

        Returns:
            A string, despite the annotation.
        """
        return "nope"  # type: ignore[return-value]

    with pytest.raises(ToolExecutionError, match="handler failed: boom"):
        broken.execute({"a": 1})
    with pytest.raises(ToolExecutionError, match="invalid return value"):
        wrong_type.execute({"a": 1})


def _undocumented(a: int) -> int:
    return a


def _missing_arg_doc(a: int, b: int) -> int:
    """Do something.

    Args:
        a: Documented.

    Returns:
        Something.
    """
    return a


def _missing_returns(a: int) -> int:
    """Do something.

    Args:
        a: Documented.
    """
    return a


def _nonexistent_arg(a: int) -> int:
    """Do something.

    Args:
        a: Documented.
        b: Does not exist.

    Returns:
        Something.
    """
    return a


def _vague_hint(a: Any) -> int:
    """Do something.

    Args:
        a: Anything.

    Returns:
        Something.
    """
    return 1


def _no_return_hint(a: int):  # type: ignore[no-untyped-def]
    """Do something.

    Args:
        a: Documented.

    Returns:
        Something.
    """
    return a


def _var_args(*a: int) -> int:
    """Do something.

    Args:
        a: Documented.

    Returns:
        Something.
    """
    return 1


def _dict_param(a: dict[str, int]) -> int:
    """Do something.

    Args:
        a: An arbitrary map.

    Returns:
        Something.
    """
    return 1


async def _async_gen_tool(a: int) -> Any:
    """Do something.

    Args:
        a: Documented.

    Returns:
        Something.
    """
    yield a


@pytest.mark.parametrize(
    ("function", "message"),
    [
        (_undocumented, "missing docstring"),
        (_missing_arg_doc, "missing Args: documentation for 'b'"),
        (_missing_returns, "missing Returns"),
        (_nonexistent_arg, "nonexistent parameters"),
        (_vague_hint, "vague type hint"),
        (_no_return_hint, "return type annotation"),
        (_var_args, "keyword-compatible"),
        (_dict_param, "additionalProperties=false"),
        (_async_gen_tool, "async generators"),
    ],
)
def test_invalid_definitions_fail_at_decoration(function: Any, message: str) -> None:
    with pytest.raises(ToolDefinitionError, match=message):
        Tool(function)


def test_invalid_tool_name() -> None:
    with pytest.raises(ToolDefinitionError, match="Invalid tool name"):
        Tool(name="has space")(_missing_returns)


def test_tool_definition_error_is_a_type_error() -> None:
    assert issubclass(ToolDefinitionError, TypeError)


def test_async_tool_runs_with_aexecute_only() -> None:
    @Tool
    async def double(a: int) -> int:
        """Double a number.

        Args:
            a: The number.

        Returns:
            Twice the number.
        """
        return 2 * a

    assert double.is_async
    assert asyncio.run(double.aexecute({"a": 4})) == "8"
    with pytest.raises(ToolExecutionError, match="async tool"):
        double.execute({"a": 4})
    with pytest.raises(ToolExecutionError, match="invalid tool inputs"):
        asyncio.run(double.aexecute({"a": "x"}))


def test_sync_tool_runs_with_aexecute_in_a_thread() -> None:
    result = asyncio.run(
        get_forecast.aexecute({"location": {"city": "Wien", "country": "AT"}, "day": "2026-10-27"})
    )
    assert json.loads(result) == {"temperature": 21.5, "unit": "celsius"}
