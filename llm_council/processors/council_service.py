import jsonlines
from collections import defaultdict
from llm_council.constants import LLM_COUNCIL_MEMBERS
from llm_council.providers.base_provider import BaseProvider
from llm_council.providers.base_provider import get_provider_instance_for_llm
from llm_council.utils.jsonl_io import append_to_jsonl
from llm_council.processors.any_processor import run_processors_for_request_files


class CouncilService:
    """A service that manages the council members and their services."""

    def __init__(
        self,
        llm_council_members: list[str],
        outdir: str | None = None,
        reset_outdir: bool | None = True,
    ):
        self.llm_council_members = llm_council_members
        self.outdir = outdir

        self.council_llm_to_service_map = {
            llm: get_provider_instance_for_llm(llm) for llm in llm_council_members
        }

        if reset_outdir:
            self.reset_request_files_for_council(outdir)

    def _add_request_to_llm_services(
        self,
        llm_to_service_map: dict[str, BaseProvider],
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

    def get_response_paths(self):
        """Returns a map of llm to response path."""
        llm_to_response_path = {}
        for service in self.council_llm_to_service_map.values():
            llm_to_response_path[service.llm] = service.get_responses_path(self.outdir)
        return llm_to_response_path

    def generate_request(
        self, prompt, prompt_id, additional_metadata, temperature, schema_name
    ):
        metadata = {
            "completion_request": {
                "prompt_id": prompt_id,
                **additional_metadata,
            }
        }
        self.write_council_request(
            prompt,
            metadata,
            temperature,
            schema_name=schema_name,
        )

    def execute_council(self):
        """Executes the council and returns a map of LLM to the list of response objects."""
        request_paths = self.get_request_paths()
        run_processors_for_request_files(request_paths, self.outdir)

        # Parse the responses into a structure and return them.
        llm_to_response_path = self.get_response_paths()
        print(llm_to_response_path)
        llm_to_responses = defaultdict(list)
        for llm, response_path in llm_to_response_path.items():
            with jsonlines.open(response_path) as reader:
                for response in reader:
                    llm_to_responses[llm].append(response)
        return llm_to_responses


def get_default_council_service(outdir: str):
    """Returns a default council service based on llms defined in constants.py."""
    return CouncilService(
        llm_council_members=LLM_COUNCIL_MEMBERS,
        outdir=outdir,
    )
