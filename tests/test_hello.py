import rigload


def test_hello_comes_from_rust():
    assert rigload.hello().startswith("hello from rigload-core")
