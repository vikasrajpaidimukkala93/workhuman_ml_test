import time
import unittest
from unittest.mock import MagicMock, patch
from app.routers.utils import SimpleTTLCache, ValkeyCache

class TestSimpleTTLCache(unittest.TestCase):
    def test_set_get(self):
        cache = SimpleTTLCache(ttl=1)
        cache.set("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")

    def test_expiry(self):
        cache = SimpleTTLCache(ttl=0.1)
        cache.set("key1", "value1")
        time.sleep(0.2)
        self.assertIsNone(cache.get("key1"))

    def test_delete(self):
        cache = SimpleTTLCache(ttl=1)
        cache.set("key1", "value1")
        cache.delete("key1")
        self.assertIsNone(cache.get("key1"))

class TestValkeyCache(unittest.TestCase):
    @patch("redis.Redis")
    def test_set_get(self, mock_redis):
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        
        cache = ValkeyCache(host="localhost", port=6379, ttl=60)
        
        # Test Set
        cache.set("key1", {"a": 1})
        mock_client.set.assert_called_once()
        
        # Test Get
        mock_client.get.return_value = '{"a": 1}'
        val = cache.get("key1")
        self.assertEqual(val, {"a": 1})
        
    @patch("redis.Redis")
    def test_delete(self, mock_redis):
        mock_client = MagicMock()
        mock_redis.return_value = mock_client
        
        cache = ValkeyCache(host="localhost", port=6379, ttl=60)
        cache.delete("key1")
        mock_client.delete.assert_called_with("key1")

if __name__ == "__main__":
    unittest.main()
