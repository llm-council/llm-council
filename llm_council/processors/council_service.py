from llm_council.constants import LLM_COUNCIL_MEMBERS
from llm_council.processors.services import BaseService
from llm_council.processors.services.utils import get_service_for_llm
from llm_council.utils.jsonl_io import append_to_jsonl

from llm_council.processors.any_processor import run_processors_for_request_files


class CouncilService:
    """A service that manages the council members and their services."""

    def __init__(self, llm_council_members: list[str], outdir: str | None = None):
        self.llm_council_members = llm_council_members
        self.outdir = outdir

        self.council_llm_to_service_map = {
            llm: get_service_for_llm(llm) for llm in llm_council_members
        }

        if outdir:
            self.reset_request_files_for_council(outdir)

    def _add_request_to_llm_services(
        self,
        llm_to_service_map: dict[str, BaseService],
        metadata: dict,
        prompt: str,
        temperature: float | None,
        schema_name: str | None,
    ) -> None:
        for llm, llm_service in llm_to_service_map.items():
            request_body = llm_service.get_request_body(
                prompt, temperature, schema_name
            )
            request_body["metadata"] = metadata
            # Add llm to the metadata so that the council executor knows which service to use.
            request_body["metadata"]["llm"] = llm
            requests_path = llm_service.get_requests_path(self.outdir)
            append_to_jsonl(request_body, requests_path)

    def write_council_request(
        self,
        prompt: str,
        metadata: dict,
        temperature: float | None,
        schema_name: str | None,
    ):
        """For the given prompt and metadata, create a request for each council member."""
        self._add_request_to_llm_services(
            self.council_llm_to_service_map, metadata, prompt, temperature, schema_name
        )

    def write_council_request_for_llm(
        self,
        llm: str,
        prompt: str,
        metadata: dict,
        temperature: float | None,
        schema_name: str | None,
    ):
        """For the given prompt and metadata, create a request for the given council member."""
        self._add_request_to_llm_services(
            {llm: self.council_llm_to_service_map[llm]},
            metadata,
            prompt,
            temperature,
            schema_name,
        )

    def get_llm_response_string(self, processor_response: dict):
        llm = processor_response[0]["llm"]  # Comes from the processor.
        return self.council_llm_to_service_map[llm].get_response_string(
            processor_response[2]
        )

    def get_llm_request_prompt(self, processor_request: dict):
        llm = processor_request[0]["llm"]
        return self.council_llm_to_service_map[llm].get_request_prompt(
            processor_request[1]
        )

    def get_llm_response_query_info(self, processor_response: dict):
        llm = processor_response[0]["llm"]  # Comes from the processor.
        return self.council_llm_to_service_map[llm].get_response_info(
            processor_response[2]
        )

    def reset_request_files_for_council(self, outdir):
        for service in self.council_llm_to_service_map.values():
            service.reset_requests(outdir)

    def get_request_paths(self):
        request_paths = set()
        for service in self.council_llm_to_service_map.values():
            request_paths.add(service.get_requests_path(self.outdir))
        return sorted(list(request_paths))

    def execute_council(self):
        request_paths = self.get_request_paths()
        run_processors_for_request_files(request_paths, self.outdir)


def get_default_council_service(outdir: str):
    """Returns a default council service based on llms defined in constants.py."""
    return CouncilService(
        llm_council_members=LLM_COUNCIL_MEMBERS,
        outdir=outdir,
    )
