"""Pydantic output formats that generated agents may use, registered by the developer."""

from __future__ import annotations

from collections.abc import Iterable
import inspect

from pydantic import BaseModel

from .errors import RegistryError

__all__ = ["OutputFormatRegistry"]


class OutputFormatRegistry:
    """Maps names to Pydantic models. The Agent Architect can only pick from these;
    it never writes models itself."""

    def __init__(self, models: Iterable[type[BaseModel]] = ()) -> None:
        self._models: dict[str, type[BaseModel]] = {}
        for model in models:
            self.register(model)

    def register(self, model: type[BaseModel], name: str | None = None) -> None:
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise TypeError("Output formats must be Pydantic BaseModel subclasses")
        key = name or model.__name__
        if key in self._models:
            raise RegistryError(f"Output format already registered: {key}")
        self._models[key] = model

    def get(self, name: str) -> type[BaseModel]:
        try:
            return self._models[name]
        except KeyError:
            raise RegistryError(f"Unknown output format: {name}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._models

    def names(self) -> list[str]:
        return list(self._models)

    def catalog(self) -> list[dict[str, str]]:
        """Name, description and field names of every format, e.g. for prompts."""
        return [
            {
                "name": name,
                "description": inspect.getdoc(model) or "",
                "fields": ", ".join(model.model_fields),
            }
            for name, model in self._models.items()
        ]
