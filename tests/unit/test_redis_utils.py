"""Unit tests for Redis utilities."""

from unittest.mock import Mock, patch

import pytest

from graflow.utils.redis_utils import (
    REDIS_AVAILABLE,
    create_redis_client,
    ensure_redis_connection_params,
    extract_redis_config,
)


class TestExtractRedisConfig:
    """Tests for extract_redis_config()."""

    def test_basic_config_extraction(self) -> None:
        """Test extracting basic connection parameters."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        # Mock Redis client
        mock_client = Mock()
        mock_client.connection_pool.connection_kwargs = {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
        }
        mock_client.decode_responses = False

        config = extract_redis_config(mock_client)

        assert config['host'] == 'localhost'
        assert config['port'] == 6379
        assert config['db'] == 0
        assert config['decode_responses'] is False

    def test_config_with_password(self) -> None:
        """Test extraction includes password when present."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        mock_client = Mock()
        mock_client.connection_pool.connection_kwargs = {
            'host': 'localhost',
            'port': 6379,
            'db': 1,
            'password': 'secret123',
            'decode_responses': True,
        }

        config = extract_redis_config(mock_client)

        assert config['password'] == 'secret123'
        assert config['decode_responses'] is True

    def test_config_with_username(self) -> None:
        """Test extraction includes username when present."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        mock_client = Mock()
        mock_client.connection_pool.connection_kwargs = {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'username': 'myuser',
            'password': 'mypass',
        }
        mock_client.decode_responses = False

        config = extract_redis_config(mock_client)

        assert config['username'] == 'myuser'
        assert config['password'] == 'mypass'

    def test_config_with_ssl(self) -> None:
        """Test extraction includes SSL parameters."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        mock_client = Mock()
        mock_client.connection_pool.connection_kwargs = {
            'host': 'secure.redis.com',
            'port': 6380,
            'db': 0,
            'ssl': True,
        }
        mock_client.decode_responses = False

        config = extract_redis_config(mock_client)

        assert config['ssl'] is True

    def test_config_with_socket_timeout(self) -> None:
        """Test extraction includes socket timeout."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        mock_client = Mock()
        mock_client.connection_pool.connection_kwargs = {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'socket_timeout': 5.0,
        }
        mock_client.decode_responses = False

        config = extract_redis_config(mock_client)

        assert config['socket_timeout'] == 5.0

    def test_omits_none_password(self) -> None:
        """Test that None password is not included in config."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        mock_client = Mock()
        mock_client.connection_pool.connection_kwargs = {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'password': None,
        }
        mock_client.decode_responses = False

        config = extract_redis_config(mock_client)

        assert 'password' not in config

    def test_redis_not_available(self) -> None:
        """Test error when Redis is not installed."""
        if REDIS_AVAILABLE:
            # Simulate unavailable Redis
            with patch('graflow.utils.redis_utils.REDIS_AVAILABLE', False):
                with pytest.raises(ImportError, match="Redis package is not available"):
                    extract_redis_config(Mock())


class TestCreateRedisClient:
    """Tests for create_redis_client()."""

    def test_basic_client_creation(self) -> None:
        """Test creating client with basic config."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        from redis import Redis

        config = {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'decode_responses': False,
        }

        with patch.object(Redis, '__init__', return_value=None) as mock_init:
            client = create_redis_client(config)

            mock_init.assert_called_once_with(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=False,
            )

    def test_client_with_password(self) -> None:
        """Test creating client with password."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        from redis import Redis

        config = {
            'host': 'localhost',
            'port': 6379,
            'db': 1,
            'password': 'secret',
            'decode_responses': True,
        }

        with patch.object(Redis, '__init__', return_value=None) as mock_init:
            client = create_redis_client(config)

            mock_init.assert_called_once_with(
                host='localhost',
                port=6379,
                db=1,
                decode_responses=True,
                password='secret',
            )

    def test_client_with_all_options(self) -> None:
        """Test creating client with all optional parameters."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        from redis import Redis

        config = {
            'host': 'redis.example.com',
            'port': 6380,
            'db': 2,
            'decode_responses': True,
            'password': 'secret',
            'username': 'user',
            'socket_timeout': 10.0,
            'ssl': True,
        }

        with patch.object(Redis, '__init__', return_value=None) as mock_init:
            client = create_redis_client(config)

            mock_init.assert_called_once_with(
                host='redis.example.com',
                port=6380,
                db=2,
                decode_responses=True,
                password='secret',
                username='user',
                socket_timeout=10.0,
                ssl=True,
            )

    def test_uses_defaults(self) -> None:
        """Test that missing config values use defaults."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        from redis import Redis

        config = {}  # Empty config

        with patch.object(Redis, '__init__', return_value=None) as mock_init:
            client = create_redis_client(config)

            mock_init.assert_called_once_with(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=False,
            )

    def test_reuses_existing_redis_client(self) -> None:
        """Test that existing redis_client is returned directly."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        from redis import Redis

        mock_client = Mock()
        mock_client.__class__ = Redis  # type: ignore # Make isinstance check pass

        config = {'redis_client': mock_client}

        result = create_redis_client(config)

        # Should return the same instance
        assert result is mock_client

    def test_validates_existing_redis_client_type(self) -> None:
        """Test that invalid redis_client type raises AssertionError."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        config = {'redis_client': "not_a_redis_client"}

        with pytest.raises(AssertionError, match="Expected Redis instance"):
            create_redis_client(config)

    def test_redis_not_available(self) -> None:
        """Test error when Redis is not installed."""
        if REDIS_AVAILABLE:
            # Simulate unavailable Redis
            with patch('graflow.utils.redis_utils.REDIS_AVAILABLE', False):
                with pytest.raises(ImportError, match="Redis package is not available"):
                    create_redis_client({})


class TestEnsureRedisConnectionParams:
    """Tests for ensure_redis_connection_params()."""

    def test_ensures_connection_params_inplace(self) -> None:
        """Test that redis_client connection params are ensured in-place."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        from redis import Redis
        mock_client = Mock()
        mock_client.__class__ = Redis  # type: ignore # Make isinstance check pass
        mock_client.connection_pool.connection_kwargs = {
            'host': 'redis.example.com',
            'port': 6380,
            'db': 2,
            'password': 'mypass',
            'decode_responses': True,
        }

        config = {
            'redis_client': mock_client,
            'key_prefix': 'graflow',
        }

        # Ensure connection params
        ensure_redis_connection_params(config)

        # Should contain both redis_client and extracted params
        assert 'redis_client' in config  # Still present
        assert config['redis_client'] is mock_client
        assert config['host'] == 'redis.example.com'
        assert config['port'] == 6380
        assert config['db'] == 2
        assert config['password'] == 'mypass'
        assert config['decode_responses'] is True
        assert config['key_prefix'] == 'graflow'

    def test_does_not_overwrite_existing_params(self) -> None:
        """Test that existing params are not overwritten."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        from redis import Redis
        mock_client = Mock()
        mock_client.__class__ = Redis  # type: ignore # Make isinstance check pass
        mock_client.connection_pool.connection_kwargs = {
            'host': 'redis.example.com',
            'port': 6380,
            'db': 2,
            'decode_responses': True,
        }

        config = {
            'redis_client': mock_client,
            'host': 'custom.host.com',  # Already present - should not be overwritten
            'key_prefix': 'graflow',
        }

        ensure_redis_connection_params(config)

        # Should keep custom host
        assert config['host'] == 'custom.host.com'
        # But add other params
        assert config['port'] == 6380
        assert config['db'] == 2

    def test_no_redis_client_does_nothing(self) -> None:
        """Test that function does nothing if redis_client not present."""
        config = {
            'host': 'localhost',
            'port': 6379,
            'key_prefix': 'test',
        }
        original_config = config.copy()

        ensure_redis_connection_params(config)

        assert config == original_config

    def test_invalid_redis_client_raises_assertion_error(self) -> None:
        """Test that invalid redis_client raises AssertionError."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        config = {
            'redis_client': "not_a_redis_client",  # Invalid type
            'key_prefix': 'test',
        }

        with pytest.raises(AssertionError, match="Expected Redis instance"):
            ensure_redis_connection_params(config)

    def test_extraction_failure_raises_value_error(self) -> None:
        """Test that extraction failure raises ValueError."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        from redis import Redis
        # Mock client that will fail extraction
        mock_client = Mock()
        mock_client.__class__ = Redis  # type: ignore # Make isinstance check pass
        mock_client.connection_pool.connection_kwargs = None  # Will cause error

        config = {'redis_client': mock_client}

        with pytest.raises(ValueError, match="Failed to extract Redis connection parameters"):
            ensure_redis_connection_params(config)


class TestRoundTrip:
    """Integration tests for extract->create round trip."""

    def test_extract_and_recreate(self) -> None:
        """Test that we can extract config and recreate equivalent client."""
        if not REDIS_AVAILABLE:
            pytest.skip("Redis not available")

        from redis import Redis

        # Create a real Redis client (won't actually connect)
        original_client = Redis(
            host='redis.example.com',
            port=6380,
            db=2,
            password='secret',
            decode_responses=True,
        )

        # Extract config
        config = extract_redis_config(original_client)

        # Recreate client
        with patch.object(Redis, '__init__', return_value=None) as mock_init:
            new_client = create_redis_client(config)

            # Verify same parameters used
            mock_init.assert_called_once_with(
                host='redis.example.com',
                port=6380,
                db=2,
                decode_responses=True,
                password='secret',
            )
