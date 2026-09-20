"""Base adapter types for realtime SLAM / occupancy engines."""

from dataclasses import dataclass
from typing import Any

from selfsuvis.pipeline.core.docker import DockerImageRef


@dataclass(frozen=True)
class EngineDescriptor:
    name: str
    api_url: str
    role: str
    provider: str = "selfsuvis"
    open_source: bool = True
    service_name: str = ""
    image: DockerImageRef = DockerImageRef("")
    hardware_profile: str = ""
    required_modalities: tuple[str, ...] = ()
    recommended_modalities: tuple[str, ...] = ()
    pros: tuple[str, ...] = ()
    cons: tuple[str, ...] = ()
    integration_doc: str = ""
    notes: str = ""


class RealtimeEngineAdapter:
    descriptor: EngineDescriptor

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def api_url(self) -> str:
        return self.descriptor.api_url

    @property
    def configured(self) -> bool:
        return bool(self.api_url)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.descriptor.name,
            "role": self.descriptor.role,
            "provider": self.descriptor.provider,
            "open_source": self.descriptor.open_source,
            "service_name": self.descriptor.service_name,
            "api_url": self.descriptor.api_url,
            **self.descriptor.image.describe(),
            "hardware_profile": self.descriptor.hardware_profile,
            "required_modalities": list(self.descriptor.required_modalities),
            "recommended_modalities": list(self.descriptor.recommended_modalities),
            "pros": list(self.descriptor.pros),
            "cons": list(self.descriptor.cons),
            "integration_doc": self.descriptor.integration_doc,
            "notes": self.descriptor.notes,
        }


def build_descriptor(
    *,
    name: str,
    api_url: str,
    role: str,
    provider: str = "selfsuvis",
    open_source: bool = True,
    service_name: str = "",
    image: DockerImageRef | None = None,
    hardware_profile: str = "",
    required_modalities: tuple[str, ...] = (),
    recommended_modalities: tuple[str, ...] = (),
    pros: tuple[str, ...] = (),
    cons: tuple[str, ...] = (),
    integration_doc: str = "",
    notes: str = "",
) -> EngineDescriptor:
    return EngineDescriptor(
        name=name,
        api_url=api_url,
        role=role,
        provider=provider,
        open_source=open_source,
        service_name=service_name,
        image=image or DockerImageRef(""),
        hardware_profile=hardware_profile,
        required_modalities=required_modalities,
        recommended_modalities=recommended_modalities,
        pros=pros,
        cons=cons,
        integration_doc=integration_doc,
        notes=notes,
    )


def instantiate_adapter(
    registry: dict[str, type[RealtimeEngineAdapter]],
    name: str,
    *,
    default_name: str = "stub",
) -> RealtimeEngineAdapter:
    normalized = str(name or default_name).strip().lower()
    adapter_cls = registry.get(normalized) or registry[default_name]
    return adapter_cls()


def available_backend_urls(registry: dict[str, type[RealtimeEngineAdapter]]) -> dict[str, str]:
    return {name: adapter_cls().api_url for name, adapter_cls in registry.items()}


def describe_backends(
    registry: dict[str, type[RealtimeEngineAdapter]],
) -> dict[str, dict[str, object]]:
    return {name: adapter_cls().describe() for name, adapter_cls in registry.items()}
