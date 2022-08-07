import flax
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp


class Glove(nn.Module):
    """A simple embedding model based on gloVe.
       https://nlp.stanford.edu/projects/glove/
    """
    num_embeddings: int = 1024
    features: int = 64
    
    def setup(self):
        self._token_embedding = nn.Embed(self.num_embeddings,
                                         self.features)
        self._bias = nn.Embed(
            self.num_embeddings, 1, embedding_init=flax.linen.initializers.zeros)

    def __call__(self, inputs):
        """Calculates the approximate log count between tokens 1 and 2.

        Args:
          A batch of (token1, token2) integers representing co-occurence.

        Returns:
          Approximate log count between x and y.
        """
        token1, token2 = inputs
        embed1 = self._token_embedding(token1)
        bias1 = self._bias(token1)
        embed2 = self._token_embedding(token2)
        bias2 = self._bias(token2)
        dot_vmap = jax.vmap(jnp.dot, in_axes=[0, 0], out_axes=0)
        dot = dot_vmap(embed1, embed2)
        output = dot + bias1 + bias2
        return output

    def score_all(self, token):
        """Finds the score of token vs all tokens.

        Args:
          max_count: The maximum count of tokens to return.
          token: Integer index of token to find neighbors of.

        Returns:
          Scores of nearest tokens.
        """
        embed1 = self._token_embedding(token)
        all_tokens = jnp.arange(0, self.num_embeddings, 1, dtype=jnp.int32)
        all_embeds = self._token_embedding(all_tokens)
        dot_vmap = jax.vmap(jnp.dot, in_axes=[None, 0], out_axes=0)
        scores = dot_vmap(embed1, all_embeds)
        return scores
