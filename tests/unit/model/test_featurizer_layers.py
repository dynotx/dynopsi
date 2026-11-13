import pytest
import torch

from dynopsi.model.nn.featurization_layers import (
    ChainBreakEncoder,
    DistogramEncoder,
    IndexEmbedder,
    PairCategoricalFeatureEmbedder,
    PolymerBreakEncoder,
    SeqIdxRelativePositionEncoder,
)
from dynopsi.model.nn.layers import AdaptiveLayerNorm


@pytest.fixture
def test_params() -> dict:
    """Common test parameters used across test cases."""
    return {"batch_size": 2, "seq_len": 10, "num_atoms": 2, "k": 15, "num_distance_bins": 8, "idx_emb_dim": 4}


class TestDistogramEncoder:
    def test_distogram_encoder(self, test_params: dict) -> None:
        """Test that the distogram encoder works independently for each atom"""
        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        encoder_multi_atom = DistogramEncoder(num_distance_bins=24, num_atoms=test_params["num_atoms"])
        encoder_single_atom = DistogramEncoder(num_distance_bins=24, num_atoms=1)

        distogram_1 = encoder_multi_atom(coords)
        # Extract just within atom type distograms
        distogram_1 = distogram_1.reshape(
            test_params["batch_size"],
            test_params["seq_len"],
            test_params["seq_len"],
            test_params["num_atoms"],
            test_params["num_atoms"],
            24,
        )
        distogram_1 = torch.cat([distogram_1[:, :, :, i, i, :] for i in range(test_params["num_atoms"])], dim=-1)

        distogram_2 = torch.cat(
            [encoder_single_atom(coords[..., i : i + 1, :]) for i in range(test_params["num_atoms"])], dim=-1
        )
        assert distogram_1.shape == distogram_2.shape
        assert (distogram_1 == distogram_2).all(), "Distogram head should work independently for each atom"

    def test_symmetric_output(self, test_params: dict) -> None:
        """Test that the distogram is symmetric (d_ij = d_ji)"""
        coords = torch.randn(test_params["batch_size"], test_params["seq_len"], test_params["num_atoms"], 3)
        encoder = DistogramEncoder(num_distance_bins=10, num_atoms=test_params["num_atoms"])

        distogram = encoder(coords)  # [2, 10, 10, 2 * 2 * 10]

        # "Naive symmetry" on residue dimensions into flattened distogram should fail because atom indices must also be transposed
        assert not torch.allclose(distogram, distogram.transpose(1, 2)), (
            "Distogram should only be symmetric when (residue, atom) dimensions are jointly transposed"
        )

        distogram = distogram.reshape(
            test_params["batch_size"],
            test_params["seq_len"],
            test_params["seq_len"],
            test_params["num_atoms"],
            test_params["num_atoms"],
            -1,
        )
        assert torch.allclose(distogram, distogram.transpose(1, 2).transpose(3, 4)), (
            "Distogram should be symmetric when (residue, atom) dimensions are jointly transposed"
        )


class TestPairRelativePositionEmbedder:
    def test_pair_relative_position_embedder_symmetric(self, test_params: dict) -> None:
        """Test symmetric version of the pair relative position embedder"""
        num_bins = 31
        embedder = SeqIdxRelativePositionEncoder(num_bins=num_bins, symmetric=True)

        indices = torch.tensor([[1, 2, 3, 4]])  # batch_size=1, seq_len=4

        embeddings = embedder(indices)  # [1, 4, 4, num_bins]

        assert embeddings.shape == (1, 4, 4, num_bins), "Output shape incorrect"

        # Check symmetry property: embedding(i,j) should equal embedding(j,i)
        assert torch.allclose(embeddings[:, 0, 1], embeddings[:, 1, 0]), (
            "Symmetric embeddings should be equal for (i,j) and (j,i)"
        )

        # Check that relative positions are consistent
        # Distance between positions 1->3 should equal distance between 2->4
        assert torch.allclose(embeddings[:, 0, 2], embeddings[:, 1, 3]), (
            "Relative position embeddings should be translation invariant"
        )

    def test_pair_relative_position_embedder_non_symmetric(self, test_params: dict) -> None:
        """Test non-symmetric version of the pair relative position embedder"""
        num_bins = 31
        embedder = SeqIdxRelativePositionEncoder(num_bins=num_bins, symmetric=False)

        indices = torch.tensor([[1, 2, 3, 4]])  # batch_size=1, seq_len=4

        embeddings = embedder(indices)  # [1, 4, 4, num_bins]

        assert embeddings.shape == (1, 4, 4, num_bins), "Output shape incorrect"

        # Check non-symmetry: embedding(i,j) should NOT equal embedding(j,i)
        assert not torch.allclose(embeddings[:, 0, 1], embeddings[:, 1, 0]), (
            "Non-symmetric embeddings should differ for (i,j) and (j,i)"
        )

        # Check that forward relative positions are consistent
        # Distance from 1->3 should equal distance from 2->4
        assert torch.allclose(embeddings[:, 0, 2], embeddings[:, 1, 3]), (
            "Forward relative position embeddings should be translation invariant"
        )

        # Check that backward relative positions are consistent
        # Distance from 3->1 should equal distance from 4->2
        assert torch.allclose(embeddings[:, 2, 0], embeddings[:, 3, 1]), (
            "Backward relative position embeddings should be translation invariant"
        )

    def test_batch_independence(self, test_params: dict) -> None:
        """Test that embeddings are independent across batch dimension"""
        num_bins = 31
        embedder = SeqIdxRelativePositionEncoder(num_bins=num_bins)

        # Create two identical sequences in a batch
        indices = torch.tensor([[1, 2, 3], [1, 2, 3]])  # batch_size=2, seq_len=3

        embeddings = embedder(indices)  # [2, 3, 3, num_bins]

        # Check that embeddings are identical across batch dimension
        assert torch.allclose(embeddings[0], embeddings[1]), (
            "Embeddings should be identical for identical sequences across batch"
        )

    def test_pair_relative_position_embedder_asym_id(self, test_params: dict) -> None:
        """Test the pair relative position embedder with multiple chains/asym_ids"""
        num_bins = 31
        embedder = SeqIdxRelativePositionEncoder(num_bins=num_bins, symmetric=False)

        indices = torch.tensor([[1, 2, 3, 4]])  # batch_size=1, seq_len=4
        asym_id_same = torch.tensor([[1, 1, 1, 1]])  # batch_size=1, seq_len=4
        asym_id_diff = torch.tensor([[0, 0, 1, 1]])  # batch_size=1, seq_len=4

        embeddings_same = embedder(indices, asym_id_same)  # [1, 4, 4, num_bins]
        embeddings_diff = embedder(indices, asym_id_diff)  # [1, 4, 4, num_bins]

        # Between chain relative positions should be different
        assert not torch.allclose(embeddings_same, embeddings_diff), (
            "Single and multi-chain embeddings should differ for the inter-chain relative positions"
        )

        # Within chain relative positions should be consistent
        assert torch.allclose(embeddings_same[:, :2, :2], embeddings_diff[:, :2, :2]), (
            "Within chain relative positions should be consistent"
        )
        assert torch.allclose(embeddings_same[:, 2:, 2:], embeddings_diff[:, 2:, 2:]), (
            "Within chain relative positions should be consistent"
        )

        # For multi-chain embeddings, between chain relative positions should be set to max
        buckets_diff = embeddings_diff.argmax(dim=-1)
        assert torch.all(buckets_diff[:, :2, 2:] == 0), (
            "Between chain relative positions should be set to (negative) max distance bucket"
        )
        assert torch.all(buckets_diff[:, 2:, :2] == 30), (
            "Between chain relative positions should be set to (positive) max distance bucket"
        )

    def test_pair_relative_position_min_max_bins(self, test_params: dict) -> None:
        """Test the pair relative position embedder with multiple chains/asym_ids"""
        num_bins = 5
        embedder = SeqIdxRelativePositionEncoder(num_bins=num_bins, symmetric=False)
        embedder_symmetric = SeqIdxRelativePositionEncoder(num_bins=num_bins, symmetric=True)

        indices = torch.tensor([[1, 8]])  # batch_size=1, seq_len=4

        # The following will fails if num_bins is not set correctly in bin limits calculation.
        embeddings = embedder(indices)
        embeddings_symmetric = embedder_symmetric(indices)

        expected_bins = torch.tensor([[2, 0], [4, 2]])
        expected_bins_symmetric = torch.tensor([[0, 4], [4, 0]])

        assert torch.allclose(embeddings.argmax(dim=-1), expected_bins), (
            "Edge case bins are not handled correctly in asymmetric relative position embedder case"
        )
        assert torch.allclose(embeddings_symmetric.argmax(dim=-1), expected_bins_symmetric), (
            "Edge case bins are not handled correctly in symmetric relative position embedder case"
        )


class TestIndexEmbedder:
    def test_index_embedder(self, test_params: dict) -> None:
        embed_dim = 32
        embedder = IndexEmbedder(embedding_dim=embed_dim)

        # Create two sequences of different lengths but with shared indices
        seq1 = torch.tensor([[1, 2, 3]])  # batch_size=1, seq_len=3
        seq2 = torch.tensor([[1, 2, 3, 4, 5]])  # batch_size=1, seq_len=5

        # Get embeddings for both sequences
        emb1 = embedder(seq1)  # [1, 3, embed_dim]
        emb2 = embedder(seq2)  # [1, 5, embed_dim]

        assert emb1.shape == (1, 3, embed_dim)
        assert emb2.shape == (1, 5, embed_dim)

        # Check that embeddings for shared indices (1,2,3) are equal
        assert torch.allclose(emb1[:, :3], emb2[:, :3]), "Embeddings for shared indices should be equal"

    def test_batch_independence(self) -> None:
        embed_dim = 32
        embedder = IndexEmbedder(embedding_dim=embed_dim)

        # Create a batch with two identical sequences
        batch = torch.tensor([[1, 2, 3], [1, 2, 3]])  # batch_size=2, seq_len=3
        emb = embedder(batch)  # [2, 3, embed_dim]

        # Check that embeddings are identical across batch dimension
        assert torch.allclose(emb[0], emb[1]), "Embeddings should be identical for identical sequences across batch"

    def test_zero_index(self) -> None:
        embed_dim = 32
        embedder = IndexEmbedder(embedding_dim=embed_dim)

        # Test with zero index which is a common edge case
        indices = torch.tensor([[0, 1, 2]])
        emb = embedder(indices)

        # Zero index should produce a valid, unique embedding
        assert not torch.allclose(emb[:, 0], emb[:, 1]), "Zero index should have unique embedding"
        assert not torch.all(emb[:, 0] == 0), "Zero index should not produce all-zero embedding"


class TestAdaptiveLayerNorm:
    def test_adaptive_layer_norm_shape(self) -> None:
        """Test basic shape and masking behavior"""
        x = torch.randn(10, 10, 32)
        cond = torch.randn(10, 10, 32)
        mask = torch.randint(0, 2, (10, 10))
        norm = AdaptiveLayerNorm(dim=32, dim_cond=32)
        output = norm(x, cond, mask)
        assert output.shape == (10, 10, 32)

    def test_batch_independence(self) -> None:
        """Test that batch computation matches individual forward passes"""
        batch_size = 2
        seq_len = 10
        dim = 32

        # Initialize layer
        norm = AdaptiveLayerNorm(dim=dim, dim_cond=dim)

        # Create batch inputs
        x = torch.randn(batch_size, seq_len, seq_len, dim)
        cond = torch.randn(batch_size, seq_len, seq_len, dim)

        # Get output for full batch
        batch_output = norm(x, cond)

        # Get output for individual samples
        output1 = norm(x[0:1], cond[0:1])
        output2 = norm(x[1:2], cond[1:2])

        # Concatenate individual results
        individual_concat = torch.cat([output1, output2], dim=0)

        # Verify numerical equality
        assert torch.allclose(batch_output, individual_concat, atol=1e-6), (
            "Batch computation differs from individual computations"
        )


class TestChainBreakEncoder:
    def test_chain_break_encoder_single_chain(self, test_params: dict) -> None:
        embedder = ChainBreakEncoder()

        # Create test chain indices
        single_chain_ids = torch.zeros([test_params["batch_size"], test_params["seq_len"]], dtype=torch.long)

        # Get encoding
        embeddings = embedder(single_chain_ids)

        assert embeddings.shape == (test_params["batch_size"], test_params["seq_len"]), "Output shape incorrect"

        # Check there are no chain breaks
        expected_embeddings = torch.zeros_like(embeddings)
        expected_embeddings[:, -1] = 1
        assert torch.all(embeddings == expected_embeddings), (
            "For single chain, chain break encoding should be 0 everywhere except final position"
        )

        # Check embedding doesn't change with arbitrary chain_id value shift
        chain_ids_shifted = single_chain_ids + torch.tensor([10, 42]).unsqueeze(-1)
        embeddings_shifted = embedder(chain_ids_shifted)
        assert torch.allclose(embeddings, embeddings_shifted), (
            "Chain break encoding should not change with arbitrary chain_id value shift"
        )

    def test_chain_break_encoder_multi_chain(self, test_params: dict) -> None:
        embedder = ChainBreakEncoder()

        seq_len = test_params["seq_len"]
        batch_size = 4
        # Create test chain indices
        multi_chain_ids = torch.stack(
            [
                torch.arange(seq_len),
                torch.tensor([0] * 4 + [1] * 4 + [0] * (seq_len - 8)),
                torch.tensor([0] * 4 + [2] * 4 + [0] * (seq_len - 8)),
                torch.tensor([0] * 4 + [1] * 4 + [0] * (seq_len - 8)),
            ]
        )

        resolved_mask = torch.stack(
            [
                torch.tensor([1] * seq_len),
                torch.tensor([1] * 8 + [0] * (seq_len - 8)),
                torch.tensor([0, 1, 1, 1] + [0, 1, 1, 0] + [0] * (seq_len - 8)),
                torch.tensor([1, 0, 1, 0] + [0, 1, 1, 0] + [0] * (seq_len - 8)),
            ]
        )

        expected_chain_breaks_unmasked = torch.stack(
            [
                torch.ones(10, dtype=torch.long),
                torch.tensor([0] * 3 + [1] + [0] * 3 + [1] + [0, 1]),
                torch.tensor([0, 0, 0, 1] + [0, 0, 0, 1] + [0, 1]),
                torch.tensor([0, 0, 0, 1] + [0, 0, 0, 1] + [0, 1]),
            ]
        )

        expected_chain_breaks_masked = torch.stack(
            [
                torch.ones(10, dtype=torch.long),
                torch.tensor([0] * 3 + [1] + [0] * 3 + [1] + [0, 0]),
                torch.tensor([0, 0, 0, 1] + [0, 0, 1, 0] + [0, 0]),
                torch.tensor([0, 0, 1, 0] + [0, 0, 1, 0] + [0, 0]),
            ]
        )

        # Get unmasked encoding
        embeddings_unmasked = embedder(multi_chain_ids)

        assert embeddings_unmasked.shape == (batch_size, seq_len), "Output shape incorrect"

        # Check expected output for unmasked case
        assert torch.allclose(embeddings_unmasked, expected_chain_breaks_unmasked), (
            "Chain break encoding for multi-chain case is incorrect"
        )

        # Check embedding doesn't change with arbitrary chain_id value shift
        multi_chain_ids_shifted = multi_chain_ids + torch.tensor([10, 42, 17, 27]).unsqueeze(-1)
        embeddings_shifted = embedder(multi_chain_ids_shifted)
        assert torch.allclose(embeddings_unmasked, embeddings_shifted), (
            "Chain break encoding should not change with arbitrary chain_id value shift"
        )

        # Get masked encoding
        embeddings_masked = embedder(multi_chain_ids, mask=resolved_mask)

        assert embeddings_masked.shape == (batch_size, seq_len), "Output shape incorrect"

        # Check expected output for masked case
        assert torch.allclose(embeddings_masked, expected_chain_breaks_masked), (
            "Chain break encoding for multi-chain case is incorrect"
        )

        # Check embedding doesn't change with arbitrary chain_id value shift
        multi_chain_ids_shifted = multi_chain_ids + torch.tensor([10, 42, 17, 27]).unsqueeze(-1)
        embeddings_masked_shifted = embedder(multi_chain_ids_shifted, mask=resolved_mask)
        assert torch.allclose(embeddings_masked, embeddings_masked_shifted), (
            "Chain break encoding should not change with arbitrary chain_id value shift"
        )

    def test_batch_independence(self, test_params: dict) -> None:
        """Test that embeddings are independent across batch dimension"""
        embedder = ChainBreakEncoder()
        # Create two indices tensors with same chain breaks
        indices = torch.tensor([[0, 0, 1, 1], [5, 5, 10, 10]])  # batch_size=2, seq_len=4
        embeddings = embedder(indices)  # [2, 4, 1]
        # Check that embeddings are identical across batch dimension
        assert torch.allclose(embeddings[0], embeddings[1]), (
            "Embeddings should be identical for identical sequences across batch"
        )


class TestPolymerBreakEncoder:
    def test_polymer_break_encoder(self, test_params: dict) -> None:
        embedder = PolymerBreakEncoder()

        seq_len = test_params["seq_len"]
        batch_size = 4
        # Create test chain indices

        residue_indices = torch.stack(
            [
                torch.arange(seq_len),
                torch.arange(seq_len),
                torch.tensor(list(range(0, 4)) + list(range(6, seq_len + 2))),
                torch.tensor(list(range(0, 4)) + list(range(0, seq_len - 4))),
            ]
        )
        residue_indices_shifted = residue_indices + torch.tensor([10, 42, 17, 27]).unsqueeze(-1)

        resolved_mask = torch.stack(
            [
                torch.tensor([1] * seq_len),
                torch.tensor([1, 1, 1, 0] + [1, 1, 1, 1] + [0] * (seq_len - 8)),
                torch.tensor([0, 1, 1, 1] + [1, 1, 1, 0] + [1] * (seq_len - 8)),
                torch.tensor([1, 0, 1, 1] + [1, 1, 1, 0] + [1] * (seq_len - 8)),
            ]
        )

        expected_breaks_unmasked = torch.stack(
            [
                torch.tensor([0] * (seq_len - 1) + [1], dtype=torch.long),
                torch.tensor([0] * (seq_len - 1) + [1], dtype=torch.long),
                torch.tensor([0, 0, 0, 1] + [0] * (seq_len - 5) + [1], dtype=torch.long),
                torch.tensor([0, 0, 0, 1] + [0] * (seq_len - 5) + [1], dtype=torch.long),
            ]
        )

        expected_breaks_masked = torch.stack(
            [
                torch.tensor([0] * (seq_len - 1) + [1], dtype=torch.long),
                torch.tensor([0, 0, 1, 0] + [0, 0, 0, 1] + [0] * (seq_len - 8), dtype=torch.long),
                torch.tensor([0, 0, 0, 1] + [0, 0, 1, 0] + [0] * (seq_len - 9) + [1], dtype=torch.long),
                torch.tensor([1, 0, 0, 1] + [0, 0, 1, 0] + [0] * (seq_len - 9) + [1], dtype=torch.long),
            ]
        )

        # Get unmasked encoding
        embeddings_unmasked = embedder(residue_indices)

        assert embeddings_unmasked.shape == (batch_size, seq_len), "Output shape incorrect"
        # Check expected output for unmasked case
        assert torch.allclose(embeddings_unmasked, expected_breaks_unmasked), (
            "Polymer break encoding for multi-chain unmasked case is incorrect"
        )

        # Get masked encoding
        embeddings_masked = embedder(residue_indices, mask=resolved_mask)

        assert embeddings_masked.shape == (batch_size, seq_len), "Output shape incorrect"

        assert torch.allclose(embeddings_masked, expected_breaks_masked), (
            "Polymer break encoding for multi-chain masked case is incorrect"
        )

        embeddings_shifted_unmasked = embedder(residue_indices_shifted)

        assert torch.allclose(embeddings_shifted_unmasked, expected_breaks_unmasked), (
            "Polymer break encoding should not change with arbitrary uniform residue index shift"
        )

        embeddings_shifted_masked = embedder(residue_indices_shifted, mask=resolved_mask)

        assert torch.allclose(embeddings_shifted_masked, expected_breaks_masked), (
            "Polymer break encoding should not change with arbitrary uniform residue index shift"
        )


class TestPairCategoricalFeatureEmbedder:
    def test_output_shape(self, test_params: dict) -> None:
        """Test that output shape is correct"""
        batch_size = test_params["batch_size"]
        seq_len = test_params["seq_len"]
        num_classes = 3
        embed_dim = 16

        embedder = PairCategoricalFeatureEmbedder(num_classes=num_classes, embed_dim=embed_dim)
        features = torch.randint(0, num_classes, (batch_size, seq_len))

        embeddings = embedder(features)

        assert embeddings.shape == (batch_size, seq_len, seq_len, embed_dim), "Output shape incorrect"

    def test_batch_independence(self, test_params: dict) -> None:
        """Test that embeddings are independent across batch dimension"""
        num_classes = 2
        embed_dim = 8
        embedder = PairCategoricalFeatureEmbedder(num_classes=num_classes, embed_dim=embed_dim)

        features = torch.tensor([[0, 1, 0], [0, 1, 0]])  # batch_size=2, seq_len=3

        embeddings = embedder(features)

        assert torch.allclose(embeddings[0], embeddings[1]), (
            "Embeddings should be identical for identical feature vectors across batch"
        )

        features = torch.tensor([[0, 1, 0], [0, 1, 1]])  # batch_size=2, seq_len=3

        embeddings = embedder(features)

        assert not torch.allclose(embeddings[0], embeddings[1]), (
            "Embeddings should be different for different feature vectors across batch"
        )

    def test_pair_consistency(self, test_params: dict) -> None:
        """Test that identical pairs produce identical embeddings"""
        num_classes = 3
        embed_dim = 16
        embedder = PairCategoricalFeatureEmbedder(num_classes=num_classes, embed_dim=embed_dim)

        features = torch.tensor([[0, 1, 2, 0]])  # batch_size=1, seq_len=4

        embeddings = embedder(features)

        # Pairs (0,0) at positions (0,0) and (3,3) should have identical embeddings
        assert torch.allclose(embeddings[0, 0, 0], embeddings[0, 3, 3]), (
            "Identical pairs should produce identical embeddings"
        )

        # Pairs (0,1) at positions (0,1) and (3,1) should have identical embeddings
        assert torch.allclose(embeddings[0, 0, 1], embeddings[0, 3, 1]), (
            "Identical pairs should produce identical embeddings"
        )

        # Pairs (0,1) and (0,2) should have different embeddings
        assert not torch.allclose(embeddings[0, 0, 1], embeddings[0, 0, 2]), (
            "Different pairs should produce different embeddings"
        )

        # Pairs (0, 2) and (2, 0) at positions (3,2) and (2,3) should be different
        assert not torch.allclose(embeddings[0, 3, 2], embeddings[0, 2, 3]), (
            "Pairs with different order should produce different embeddings"
        )

    def test_single_class(self, test_params: dict) -> None:
        """Test edge case with single class"""
        num_classes = 1
        embed_dim = 8
        embedder = PairCategoricalFeatureEmbedder(num_classes=num_classes, embed_dim=embed_dim)

        features = torch.zeros((2, 5), dtype=torch.long)  # batch_size=2, seq_len=5, all zeros

        embeddings = embedder(features)

        assert embeddings.shape == (2, 5, 5, embed_dim), "Output shape incorrect for single class"

        # All pairs should be identical since all features are the same
        assert torch.allclose(embeddings[0, 0, 0], embeddings[0]), "All pairs should be identical for single class"

    def test_all_pair_combinations(self, test_params: dict) -> None:
        """Test that all pair combinations are properly embedded"""
        num_classes = 2
        embed_dim = 8
        embedder = PairCategoricalFeatureEmbedder(num_classes=num_classes, embed_dim=embed_dim)

        # Create features that cover all possible pairs
        features = torch.tensor([[0, 1, 0, 1]])  # batch_size=1, seq_len=4

        embeddings = embedder(features)

        # Check that we can extract embeddings for all 4 possible pairs: (0,0), (0,1), (1,0), (1,1)
        pair_00 = embeddings[0, 0, 0]
        pair_01 = embeddings[0, 0, 1]
        pair_10 = embeddings[0, 1, 0]
        pair_11 = embeddings[0, 1, 1]

        # All pairs should be different
        assert not torch.allclose(pair_00, pair_01), "Pair (0,0) should differ from (0,1)"
        assert not torch.allclose(pair_00, pair_10), "Pair (0,0) should differ from (1,0)"
        assert not torch.allclose(pair_00, pair_11), "Pair (0,0) should differ from (1,1)"
        assert not torch.allclose(pair_01, pair_10), "Pair (0,1) should differ from (1,0)"
        assert not torch.allclose(pair_01, pair_11), "Pair (0,1) should differ from (1,1)"
        assert not torch.allclose(pair_10, pair_11), "Pair (1,0) should differ from (1,1)"
