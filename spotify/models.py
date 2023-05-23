# Copyright 2023 Hector Yee, Bryan Bischoff
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for the spotify million playlist."""

from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

class SpotifyModel(nn.Module):
    """Spotify model that takes a context and predicts the next tracks."""
    feature_size : int

    def setup(self):
        # There are too many tracks and albums so limit to this number by hashing.
        self.max_albums = 100000
        self.album_embed = nn.Embed(self.max_albums, self.feature_size)
        self.artist_embed = nn.Embed(295861, self.feature_size)

    def get_embeddings(self, album, artist):
        """
        Given track, album, artist indices return the embeddings.
        Args:
            album: ints of shape nx1
            artist: ints of shape nx1
        Returns:
            Embeddings representing the track.
        """
        album_modded = jnp.mod(album, self.max_albums)
        album_embed = self.album_embed(album_modded)
        artist_embed = self.artist_embed(artist)
        result = jnp.concatenate([album_embed, artist_embed], axis=-1)
        return result

    def __call__(self,
                 track_context, album_context, artist_context,
                 next_track, next_album, next_artist,
                 neg_track, neg_album, neg_artist):
        """Returns the affinity score to the context.
        Args:
            track_context: ints of shape n
            album_context: ints of shape n
            artist_context: ints of shape n
            next_track: int of shape m
            next_album: int of shape m
            next_artist: int of shape m
            neg_track: int of shape o
            neg_album: int of shape o
            neg_artist: int of shape o
        Returns:
            pos_affinity: the affinity of the context to the next track of shape m.
            neg_affinity: the affinity of the context to the negative tracks of shape o.
        """
        context_embed = self.get_embeddings(album_context, artist_context)
        next_embed = self.get_embeddings(next_album, next_artist)
        neg_embed = self.get_embeddings(neg_album, neg_artist)

        # The affinity of the context to the other track is simply the dot product of
        # each context embedding with the other track's embedding.
        # We also add a small boost if the album or artist match.
        pos_affinity = jnp.max(jnp.dot(next_embed, context_embed.T), axis=-1)
        pos_affinity = pos_affinity + 0.1 * jnp.isin(next_album, album_context)
        pos_affinity = pos_affinity + 0.1 * jnp.isin(next_artist, artist_context)

        neg_affinity = jnp.max(jnp.dot(neg_embed, context_embed.T), axis=-1)
        neg_affinity = neg_affinity + 0.1 * jnp.isin(neg_album, album_context)
        neg_affinity = neg_affinity + 0.1 * jnp.isin(neg_artist, artist_context)

        all_embeddings = jnp.concatenate([context_embed, next_embed, neg_embed], axis=-2)
        all_embeddings_l2 = jnp.sqrt(jnp.sum(jnp.square(all_embeddings), axis=-1))

        context_self_affinity = jnp.dot(jnp.flip(context_embed, axis=-2), context_embed.T)
        next_self_affinity = jnp.dot(jnp.flip(next_embed, axis=-2), next_embed.T)

        return pos_affinity, neg_affinity, context_self_affinity, next_self_affinity, all_embeddings_l2