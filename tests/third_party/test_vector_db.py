from pathlib import Path

import pytest

from ds_agent.rag.db_faiss import VectorFaissDB, VectorDBInfo, DB_FAISS, CodeErrorExample


@pytest.fixture(scope="session")
def vector_db(tmp_path_factory):
    db_faiss_path = tmp_path_factory.mktemp("test-vector-db")
    embedding_model_name = "thenlper/gte-small"  # "sentence-transformers/all-mpnet-base-v2"

    DB_FAISS.start(db_path=db_faiss_path, model_name=embedding_model_name, embedded_field="code", is_empty=True)
    return DB_FAISS


def test_vector_db_insert_example(vector_db):
    """Check example insertion works properly"""
    example = CodeErrorExample(
        template_file="some/code/template.jinja",
        code='```python\n"""Code that causes a failing unittest"""\n```',
        code_error="Exception: caused by code",
        code_fixed='```python\n"""Code that fixes the unittest"""\n```',
        task_id="kaggle-task-id",
    )

    # Test that inserting into non-existent db throws
    with pytest.raises(ValueError):
        vector_db.insert_code_examples([example], create_missing=False)

    # Check that the missing db can be automatically created
    vector_db.insert_code_examples([example], create_missing=True)

    # Retrieve to check it's actually inserted
    retrieved_example = vector_db.retrieve_top_template_error_example(example.code, template_file=example.template_file)
    assert example == retrieved_example

    # Test held out retrieval
    retrieved_example2 = vector_db.retrieve_top_template_error_example(
        example.code, template_file=example.template_file, heldout_metadata=dict(task_id=example.task_id)
    )
    assert retrieved_example2 is None


def test_vector_db_info_extras():
    """VectorDBInfo should allow arbitrary metadata fields"""
    VectorDBInfo(embedded_field="", model_name="", weird_stuff=123)


def test_vector_db_singleton(vector_db):
    s2 = VectorFaissDB()

    assert vector_db == s2, "Error the two objects are different"

    with pytest.raises(AssertionError):
        s2.start(model_name="thenlper/gte-small", db_path="wherever/rag_db", embedded_field="code")

