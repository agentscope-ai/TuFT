"""Authentication utilities for LLM-RPC server.

The current implementation is only for demonstration purposes and
should be replaced with a proper authentication system in the future.
Planned improvements:

1. persistent storage
2. API key hashing (store hashed key instead of actual keys in persistent storage)
3. API key format with format validation to avoid hitting db every time
4. API key expiry
"""


class User:
    """A simple user representation. Enhance it in the future."""

    def __init__(self, user_id: str):
        self.user_id = user_id


class AuthenticationDB:
    """A simple in-memory authentication database.
    It maps API keys to User instances.
    """

    def __init__(self, authorized_users: dict[str, str]):
        """Initialize the authentication database."""
        self.authorized_users = authorized_users

    def authenticate(self, api_key: str) -> User | None:
        """Authenticate a user by API key."""
        user_id = self.authorized_users.get(api_key)
        if user_id:
            return User(user_id)
        return None
