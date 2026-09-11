import pytest

from transformer.tokenizers import CharTokenizer

CORPUS = ['hello world', 'hi']
ROUND_TRIP_TEXTS = ['hello', 'hi world', '']


@pytest.fixture
def tok():
    return CharTokenizer(CORPUS)


@pytest.fixture
def ab_tok():
    return CharTokenizer(['ab'])  # vocab: <pad><start><end><unk> a b  -> a=4 b=5


class TestTokenizerConstruction:
    @pytest.mark.parametrize(
        'corpus,expected_chars',
        [
            (['cba'], ['a', 'b', 'c']),
            (['abc', 'b'], ['a', 'b', 'c']),
            (['ab a'], [' ', 'a', 'b']),
            (['aAa'], ['A', 'a']),
        ],
    )
    def test_vocab_layout(self, corpus, expected_chars):
        tok = CharTokenizer(corpus)
        assert tok.vocab == CharTokenizer.SPECIALS + expected_chars

    @pytest.mark.parametrize('corpus', [[], [''], ['', '']])
    def test_empty_corpus_raises_error(self, corpus):
        with pytest.raises(ValueError):
            CharTokenizer(corpus)

    @pytest.mark.parametrize(
        'corpus,expected_chars',
        [
            ([' '], [' ']),
            (['a'], ['a']),
        ],
    )
    def test_single_char_corpus_is_allowed(self, corpus, expected_chars):
        tok = CharTokenizer(corpus)
        assert tok.vocab == CharTokenizer.SPECIALS + expected_chars

    def test_special_ids_present(self, tok):
        assert (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id) == (0, 1, 2, 3)

    def test_vocab_order_is_deterministic(self):
        tok1 = CharTokenizer(['abc', 'def a'])
        tok2 = CharTokenizer(['a', 'fed cba'])
        assert tok1.vocab == tok2.vocab
        assert tok1.stoi == tok2.stoi

    def test_has_token(self):
        corpus = ['abc']
        corpus_chars = list(''.join(corpus))
        tok = CharTokenizer(corpus)
        for ch in CharTokenizer.SPECIALS + corpus_chars:
            assert tok._has_token(ch) is True
        assert tok._has_token('d') is False

    def test_stoi_maps_to_vocab(self):
        tok = CharTokenizer(['abc'])
        assert tok.stoi == {t: i for i, t in enumerate(tok.vocab)}

    @pytest.mark.parametrize(
        'corpus,n_distinct',
        [
            (['abc'], 3),
            (['abc', 'b'], 3),
            (['a a'], 2),
            (['aAa'], 2),
        ],
    )
    def test_vocab_size(self, corpus, n_distinct):
        tok = CharTokenizer(corpus)
        assert tok.vocab_size == len(CharTokenizer.SPECIALS) + n_distinct

    def test_special_token_chars_do_not_collide(self):
        corpus = ['<pad>']
        expected_vocab = CharTokenizer.SPECIALS + ['<', '>', 'a', 'd', 'p']

        tok = CharTokenizer(corpus)
        assert tok.vocab == expected_vocab
        assert tok.token_to_id('<pad>') != tok.token_to_id('<')


class TestEncode:
    @pytest.mark.parametrize(
        'text,expected',
        [
            ('', []),
            ('a', [4]),
            ('bab', [5, 4, 5]),  # order preserved
            ('z', [3]),  # unknown char -> unk_id
        ],
    )
    def test_encode_maps_chars_to_id(self, ab_tok, text, expected):
        assert ab_tok.encode(text) == expected

    @pytest.mark.parametrize('text', ['hi', ''])
    def test_encode_with_specials(self, tok, text):
        encoded_body = tok.encode(text, add_special_tokens=False)
        assert tok.encode(text, add_special_tokens=True) == [tok.bos_id, *encoded_body, tok.eos_id]


class TestDecode:
    @pytest.mark.parametrize(
        'ids,expected',
        [
            ([], ''),
            ([4], 'a'),
            ([5, 4, 5], 'bab'),  # order preserved
            ([1, 5, 4, 2], 'ba'),  # specials stripped by default
            ([3], ''),  # special char
            ([1], ''),  # special char
        ],
    )
    def test_decode_maps_ids_to_char(self, ab_tok, ids, expected):
        assert ab_tok.decode(ids) == expected

    def test_decode_keeps_specials(self, ab_tok):
        # vocab: <pad><start><end><unk> a b  -> <pad>=0, <start>=1, <end>=2, <unk>=3, a=4, b=5
        assert ab_tok.decode([1, 5, 4, 3, 4, 2], skip_special_tokens=False) == '<start>ba<unk>a<end>'

    @pytest.mark.parametrize('idx', [999, -1])
    def test_decode_rejects_out_of_range_id(self, ab_tok, idx):
        with pytest.raises(IndexError):
            ab_tok.decode([idx])


class TestRoundTrip:
    @pytest.mark.parametrize('text', ROUND_TRIP_TEXTS)
    def test_encode_decode_roundtrip(self, tok, text):
        assert tok.decode(tok.encode(text)) == text

    @pytest.mark.parametrize('text', ROUND_TRIP_TEXTS)
    def test_roundtrip_strips_added_specials(self, tok, text):
        assert tok.decode(tok.encode(text, add_special_tokens=True)) == text

    @pytest.mark.parametrize('text', ROUND_TRIP_TEXTS)
    def test_encode_decode_with_specials(self, tok, text):
        assert (
            tok.decode(tok.encode(text, add_special_tokens=True), skip_special_tokens=False)
            == f'{tok.BOS}{text}{tok.EOS}'
        )

    def test_unknown_char_roundtrip(self, tok):
        assert tok.decode(tok.encode('z')) == ''
        assert tok.decode(tok.encode('z'), skip_special_tokens=False) == tok.UNK

    def test_unknown_char_midstring_roundtrip(self, tok):
        assert tok.decode(tok.encode('heZo')) == 'heo'
        assert tok.decode(tok.encode('heZo'), skip_special_tokens=False) == f'he{tok.UNK}o'
        assert (
            tok.decode(tok.encode('heZo', add_special_tokens=True), skip_special_tokens=False)
            == f'{tok.BOS}he{tok.UNK}o{tok.EOS}'
        )


class TestTokenIDConversion:
    @pytest.mark.parametrize(
        'token,expected_id',
        [
            ('a', 4),
            ('<pad>', 0),
            ('<unk>', 3),
            ('A', 3),  # token not in vocab
        ],
    )
    def test_token_to_id(self, ab_tok, token, expected_id):
        assert ab_tok.token_to_id(token) == expected_id

    @pytest.mark.parametrize(
        'idx,expected_token',
        [
            (4, 'a'),
            (0, '<pad>'),
            (3, '<unk>'),
        ],
    )
    def test_id_to_token(self, ab_tok, idx, expected_token):
        assert ab_tok.id_to_token(idx) == expected_token

    @pytest.mark.parametrize('idx', [999, -1])
    def test_id_to_token_raises_error(self, ab_tok, idx):
        with pytest.raises(IndexError):
            ab_tok.id_to_token(idx)

    def test_id_to_token_boundary(self, ab_tok):
        assert ab_tok.id_to_token(ab_tok.vocab_size - 1) == 'b'
        with pytest.raises(IndexError):
            ab_tok.id_to_token(ab_tok.vocab_size)


class TestSaveLoadTokenizer:
    def test_save(self, ab_tok):
        with pytest.raises(NotImplementedError):
            ab_tok.save('temp/ab_tok_test')

    def test_load(self):
        with pytest.raises(NotImplementedError):
            CharTokenizer.load('temp/ab_tok_test')
