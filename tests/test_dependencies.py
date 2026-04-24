def test_jwt_importable():
    import jwt
    import cryptography
    assert jwt is not None
    assert cryptography is not None
