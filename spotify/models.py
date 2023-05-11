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
        self.track_embed = nn.Embed(2262292, self.feature_size)
        self.artist_embed = nn.Embed(295860, self.feature_size)
        self.album_embed = nn.Embed(734684, self.feature_size)

    def get_embeddings(self, track, artist, album):
        """
        Given track, artist and album indices return the embeddings.
        Args:
            track: ints of shape nx1
            artist: ints of shape nx1
            album: ints of shape nx1
        Returns:
            Embeddings representing the track.
        """

        track_embed = self.track_embed(track)
        artist_embed = self.artist_embed(artist)
        album_embed = self.album_embed(album)
        result = jnp.concatenate([track_embed, artist_embed, album_embed], axis=-1)
        return result

    def __call__(self,
                 track_context, artist_context, album_context,
                 next_track, next_artist, next_album):
        """Returns the mean affinity score to the context.
        Args:
            track_context: ints of shape nx1
            artist_context: ints of shape nx1
            album_context: ints of shape nx1
            next_track: int
            next_artist: int
            next_album: int
        Returns:
            mean_affinity: the mean affinity of the context to the next track.
        """
        context_embed = self.get_embeddings(track_context, artist_context, album_context)
        next_embed = self.get_embeddings(next_track, next_artist, next_album)

        # The affinity of the context to the next track is simply the dot product of
        # each context embedding with the next track's embedding.
        affinity = jnp.sum(context_embed * next_embed, axis=-1)

        # We then return the mean affinity of the context to the next track.
        mean_affinity = jnp.mean(affinity, axis = -1)
        return mean_affinity