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
        track_embed = self.track_embed(track)
        artist_embed = self.artist_embed(artist)
        album_embed = self.album_embed(album)
        return track_embed, artist_embed, album_embed

    def __call__(self,
                 track_context, artist_context, album_context,
                 next_track, next_artist, next_album):
        """Returns the maximum affinity score to the context."""
        result = self.get_embeddings(track_context, artist_context, album_context)
        track_context_embed, artist_context_embed, album_context_embed  = result
        result2 = self.get_embeddings(next_track, next_artist, next_album)
        next_track_embed, next_artist_embed, next_album_embed = result2

        # The affinity of the context to the next track is simply the dot product of
        # each context embedding with the next track's embedding.
        track_affinity = jnp.max(jnp.sum(track_context_embed * next_track_embed, axis=-1), axis=-1)
        artist_affinity = jnp.max(jnp.sum(artist_context_embed * next_artist_embed, axis=-1), axis=-1)
        album_affinity =  jnp.max(jnp.sum(album_context_embed * next_album_embed, axis=-1), axis=-1)
        return track_affinity, artist_affinity, album_affinity