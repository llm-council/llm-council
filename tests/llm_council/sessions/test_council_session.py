from llm_council.sessions.council_session import CouncilSession


def test_load_council_session():
    council_session = CouncilSession.load("tests/testdata/sample_session")
    assert council_session is not None
