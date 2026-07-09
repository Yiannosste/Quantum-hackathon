from qlkit.core.qubo import QUBO, QUBOBuilder, bits_from_string, string_from_bits


def test_energy_linear_and_quadratic():
    q = QUBOBuilder().add_linear(0, 2.0).add_linear(1, -1.0).add(0, 1, 3.0).add_offset(5.0).build()
    assert q.num_vars == 2
    assert q.energy((0, 0)) == 5.0
    assert q.energy((1, 0)) == 7.0
    assert q.energy((0, 1)) == 4.0
    assert q.energy((1, 1)) == 9.0


def test_builder_normalizes_key_order_and_merges():
    q = QUBOBuilder().add(3, 1, 2.0).add(1, 3, 0.5).build()
    assert q.terms == {(1, 3): 2.5}
    assert q.num_vars == 4


def test_addition_and_scaling():
    a = QUBOBuilder().add_linear(0, 1.0).add_offset(1.0).build()
    b = QUBOBuilder().add_linear(0, 2.0).add(0, 1, 4.0).build()
    combined = a + b
    assert combined.terms[(0, 0)] == 3.0
    assert combined.terms[(0, 1)] == 4.0
    assert combined.offset == 1.0
    assert combined.num_vars == 2

    doubled = combined.scaled(2.0)
    assert doubled.terms[(0, 1)] == 8.0
    assert doubled.offset == 2.0


def test_bitstring_round_trip():
    assert bits_from_string("0110") == (0, 1, 1, 0)
    assert string_from_bits((0, 1, 1, 0)) == "0110"
