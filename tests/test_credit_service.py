"""
Tests for services/credit_service.py — all Mongo/Redis calls are mocked
(this repo's tests run against the real dev Mongo Atlas cluster otherwise,
see tests/conftest.py, so credit deduction/lookup must never touch it for
real here).
"""
from unittest.mock import AsyncMock, patch

import pytest

from services.credit_service import CreditService, DEFAULT_CREDIT_CAP_USD


@pytest.mark.asyncio
async def test_has_credit_true_when_under_cap():
    with patch("services.credit_service.CreditService._is_admin", AsyncMock(return_value=False)), \
         patch("services.credit_service.CreditService.get_spend", AsyncMock(return_value=1.0)), \
         patch("services.credit_service.CreditService.get_cap", AsyncMock(return_value=5.0)):
        assert await CreditService.has_credit("507f1f77bcf86cd799439011") is True


@pytest.mark.asyncio
async def test_has_credit_false_when_at_or_over_cap():
    with patch("services.credit_service.CreditService._is_admin", AsyncMock(return_value=False)), \
         patch("services.credit_service.CreditService.get_spend", AsyncMock(return_value=5.0)), \
         patch("services.credit_service.CreditService.get_cap", AsyncMock(return_value=5.0)):
        assert await CreditService.has_credit("507f1f77bcf86cd799439011") is False


@pytest.mark.asyncio
async def test_has_credit_true_for_admin_even_over_cap():
    with patch("services.credit_service.CreditService._is_admin", AsyncMock(return_value=True)), \
         patch("services.credit_service.CreditService.get_spend", AsyncMock(return_value=999.0)) as mock_spend, \
         patch("services.credit_service.CreditService.get_cap", AsyncMock(return_value=5.0)) as mock_cap:
        assert await CreditService.has_credit("507f1f77bcf86cd799439011") is True
        # Admin short-circuits before spend/cap are even checked.
        mock_spend.assert_not_awaited()
        mock_cap.assert_not_awaited()


@pytest.mark.asyncio
async def test_is_admin_reads_flag_from_user_doc():
    with patch("services.credit_service.users_collection") as mock_users:
        mock_users.find_one = AsyncMock(return_value={"is_admin": True})
        assert await CreditService._is_admin("507f1f77bcf86cd799439011") is True

        mock_users.find_one = AsyncMock(return_value={"is_admin": False})
        assert await CreditService._is_admin("507f1f77bcf86cd799439011") is False

        mock_users.find_one = AsyncMock(return_value={})
        assert await CreditService._is_admin("507f1f77bcf86cd799439011") is False


@pytest.mark.asyncio
async def test_get_spend_cache_hit_skips_mongo():
    with patch("services.credit_service.cache_get", AsyncMock(return_value=2.5)) as mock_get, \
         patch("services.credit_service.users_collection") as mock_users:
        spend = await CreditService.get_spend("507f1f77bcf86cd799439011")
        assert spend == 2.5
        mock_get.assert_awaited_once()
        mock_users.find_one.assert_not_called()


@pytest.mark.asyncio
async def test_get_spend_cache_miss_falls_back_to_mongo_and_repopulates():
    with patch("services.credit_service.cache_get", AsyncMock(return_value=None)), \
         patch("services.credit_service.cache_set", AsyncMock()) as mock_set, \
         patch("services.credit_service.users_collection") as mock_users:
        mock_users.find_one = AsyncMock(return_value={"credits_used_usd": 3.25})
        spend = await CreditService.get_spend("507f1f77bcf86cd799439011")
        assert spend == 3.25
        mock_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_spend_defaults_to_zero_for_new_user():
    with patch("services.credit_service.cache_get", AsyncMock(return_value=None)), \
         patch("services.credit_service.cache_set", AsyncMock()), \
         patch("services.credit_service.users_collection") as mock_users:
        # No credits_used_usd field yet — brand new user, no migration needed.
        mock_users.find_one = AsyncMock(return_value={})
        spend = await CreditService.get_spend("507f1f77bcf86cd799439011")
        assert spend == 0.0


@pytest.mark.asyncio
async def test_get_cap_defaults_when_field_absent():
    with patch("services.credit_service.users_collection") as mock_users:
        mock_users.find_one = AsyncMock(return_value={})
        cap = await CreditService.get_cap("507f1f77bcf86cd799439011")
        assert cap == DEFAULT_CREDIT_CAP_USD


@pytest.mark.asyncio
async def test_get_cap_uses_per_user_override():
    with patch("services.credit_service.users_collection") as mock_users:
        mock_users.find_one = AsyncMock(return_value={"credit_cap_usd": 50.0})
        cap = await CreditService.get_cap("507f1f77bcf86cd799439011")
        assert cap == 50.0


@pytest.mark.asyncio
async def test_record_and_deduct_increments_mongo_atomically():
    with patch("services.credit_service.users_collection") as mock_users, \
         patch("services.credit_service.get_redis", AsyncMock(return_value=None)):
        mock_users.update_one = AsyncMock()
        await CreditService.record_and_deduct("507f1f77bcf86cd799439011", 0.0042)
        mock_users.update_one.assert_awaited_once()
        call_args = mock_users.update_one.call_args.args
        assert call_args[1] == {"$inc": {"credits_used_usd": 0.0042}}


@pytest.mark.asyncio
async def test_record_and_deduct_skips_when_cost_is_zero():
    with patch("services.credit_service.users_collection") as mock_users:
        mock_users.update_one = AsyncMock()
        await CreditService.record_and_deduct("507f1f77bcf86cd799439011", 0.0)
        mock_users.update_one.assert_not_called()


@pytest.mark.asyncio
async def test_record_and_deduct_updates_redis_best_effort():
    mock_redis = AsyncMock()
    with patch("services.credit_service.users_collection") as mock_users, \
         patch("services.credit_service.get_redis", AsyncMock(return_value=mock_redis)):
        mock_users.update_one = AsyncMock()
        await CreditService.record_and_deduct("507f1f77bcf86cd799439011", 1.5)
        mock_redis.incrbyfloat.assert_awaited_once()
        mock_redis.expire.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_and_deduct_mongo_failure_does_not_touch_redis():
    """If the durable write fails, don't let the cache silently drift ahead of it."""
    mock_redis = AsyncMock()
    with patch("services.credit_service.users_collection") as mock_users, \
         patch("services.credit_service.get_redis", AsyncMock(return_value=mock_redis)):
        mock_users.update_one = AsyncMock(side_effect=RuntimeError("mongo down"))
        await CreditService.record_and_deduct("507f1f77bcf86cd799439011", 1.5)  # must not raise
        mock_redis.incrbyfloat.assert_not_called()
