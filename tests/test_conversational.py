"""Tests for tools/conversational.py: the two canned reply kinds. No LLM
is involved at all, so there is nothing here to force offline."""

import pytest

from tools import conversational as conv


def test_greeting_reply_names_the_system_and_suggests_examples():
    result = conv.reply("greeting")
    assert result.kind == "greeting"
    assert "SatQuery AI" in result.reply
    for example in conv.EXAMPLE_QUESTIONS:
        assert example in result.reply


def test_off_topic_reply_is_scoped_and_suggests_examples():
    result = conv.reply("off_topic")
    assert result.kind == "off_topic"
    assert "only answer questions about" in result.reply.lower()
    for example in conv.EXAMPLE_QUESTIONS:
        assert example in result.reply


def test_greeting_and_off_topic_replies_are_different_text():
    greeting = conv.reply("greeting")
    off_topic = conv.reply("off_topic")
    assert greeting.reply != off_topic.reply


def test_reply_is_deterministic_not_generated():
    # Same kind, same reply every time -- a template, not a model sample.
    assert conv.reply("greeting").reply == conv.reply("greeting").reply


def test_unknown_kind_raises():
    with pytest.raises(conv.ConversationalError, match="kind"):
        conv.reply("banana")


def test_kinds_tuple_matches_what_reply_actually_accepts():
    for kind in conv.KINDS:
        result = conv.reply(kind)  # no raise
        assert result.kind == kind
