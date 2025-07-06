import dotenv

from lm_council.council import LanguageModelCouncil


def test_get_affinity():
    dotenv.load_dotenv()

    lmc = LanguageModelCouncil.load("analysis/sample_council/pairwise")
    affinity = lmc.affinity(show_plots=False)

    assert len(affinity) == 3

    lmc = LanguageModelCouncil.load("analysis/sample_council/rubric")
    affinity = lmc.affinity(show_plots=False)

    assert len(affinity) == 3


def test_get_agreement():
    dotenv.load_dotenv()

    lmc = LanguageModelCouncil.load("analysis/sample_council/pairwise")

    judge_agreement, mean_agreement_df = lmc.judge_agreement(show_plots=False)

    assert judge_agreement is not None
    assert mean_agreement_df is not None
    assert mean_agreement_df.shape[0] > 0
    assert mean_agreement_df.shape[1] > 0
