"""model_check on small hand-written files laid out like the Hugging Face cache in tmp_path."""

import json

import pytest

from silent_rag.inject.model_check import check_replacement_model

REV, BASE_REV = "a" * 40, "b" * 40
BGE_CARD = 'query_instruction = "Represent this sentence for searching relevant passages: "'
E5_CARD = 'Use "query: " for queries and "passage: " for passages.'


def write_model(cache, repo, rev, dim=384, normalize=True, dense_out=None, st_config=None, card="A model.", skip=()):
    snap = cache / f"models--{repo.replace('/', '--')}" / "snapshots" / rev
    types = ["Transformer", "Pooling"] + (["Dense"] if dense_out else []) + (["Normalize"] if normalize else [])
    paths = {"Transformer": "", "Pooling": "1_Pooling", "Dense": "2_Dense", "Normalize": "3_Normalize"}
    files = {
        "modules.json": [{"idx": i, "name": str(i), "path": paths[t], "type": f"sentence_transformers.models.{t}"}
                         for i, t in enumerate(types)],
        "1_Pooling/config.json": {"word_embedding_dimension": dim},
        "config_sentence_transformers.json": st_config or {"prompts": {}, "default_prompt_name": None},
    }
    if dense_out:
        files["2_Dense/config.json"] = {"in_features": dim, "out_features": dense_out}
    for name, content in files.items():
        if name not in skip:
            (snap / name).parent.mkdir(parents=True, exist_ok=True)
            (snap / name).write_text(json.dumps(content))
    if card is not None and "README.md" not in skip:
        (snap / "README.md").write_text(card)


@pytest.fixture
def cache(tmp_path):
    write_model(tmp_path, "base/model", BASE_REV)
    return tmp_path


def check(cache, repo="alt/model", rev=REV, base_rev=BASE_REV):
    check_replacement_model(repo, rev, "base/model", base_rev, cache_dir=cache)


def test_normalised_model_of_the_same_dimension_passes(cache):
    write_model(cache, "alt/model", REV)
    check(cache)


def test_missing_normalize_is_rejected(cache):
    write_model(cache, "alt/model", REV, normalize=False)
    with pytest.raises(ValueError, match="no Normalize module"):
        check(cache)


@pytest.mark.parametrize("kw", [{"dim": 768}, {"dense_out": 256}], ids=["pooling", "dense"])
def test_other_dimension_is_rejected(cache, kw):
    write_model(cache, "alt/model", REV, **kw)
    with pytest.raises(ValueError, match="-dim vectors, the baseline base/model gives 384"):
        check(cache)


def test_e5_style_card_with_a_passage_prefix_is_rejected(cache):
    write_model(cache, "alt/model", REV, card=E5_CARD)
    with pytest.raises(ValueError, match="'passage: ' document prefix"):
        check(cache)


def test_bge_style_query_only_instruction_is_allowed(cache):
    write_model(cache, "alt/model", REV, card=BGE_CARD,
                st_config={"prompts": {"query": "Represent this sentence for searching relevant passages: "}})
    check(cache)


@pytest.mark.parametrize("st_config, match", [
    ({"prompts": {"document": "passage: "}}, "document prompts"),
    ({"prompts": {"query": "query: "}, "default_prompt_name": "query"}, "default_prompt_name"),
])
def test_document_side_prompts_are_rejected(cache, st_config, match):
    write_model(cache, "alt/model", REV, st_config=st_config)
    with pytest.raises(ValueError, match=match):
        check(cache)


@pytest.mark.parametrize("missing", ["modules.json", "1_Pooling/config.json", "README.md"])
def test_uncached_file_is_a_clear_error(cache, missing):
    write_model(cache, "alt/model", REV, skip=(missing,))
    with pytest.raises(ValueError, match=f"alt/model@{REV}: {missing} is not in the local Hugging Face cache"):
        check(cache)


def test_uncached_baseline_is_a_clear_error(tmp_path):
    write_model(tmp_path, "alt/model", REV)
    with pytest.raises(ValueError, match="base/model@.*modules.json is not in the local Hugging Face cache"):
        check(tmp_path)


def test_file_recorded_as_absent_from_the_repo_is_optional(cache):
    write_model(cache, "alt/model", REV, skip=("config_sentence_transformers.json",))
    no_exist = cache / "models--alt--model" / ".no_exist" / REV
    no_exist.mkdir(parents=True)
    (no_exist / "config_sentence_transformers.json").touch()  # how the hub cache records "repo has no such file"
    check(cache)


def test_unpinned_baseline_resolves_through_refs(cache):
    write_model(cache, "alt/model", REV)
    refs = cache / "models--base--model" / "refs"
    refs.mkdir()
    (refs / "main").write_text(BASE_REV)
    check(cache, base_rev=None)


def test_replacement_revision_must_be_a_commit_hash(cache):
    with pytest.raises(ValueError, match="40-character lowercase commit hash"):
        check(cache, rev="main")
