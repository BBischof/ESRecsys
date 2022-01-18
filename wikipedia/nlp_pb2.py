# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: nlp.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='nlp.proto',
  package='topicspace.corpus.nlp',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\tnlp.proto\x12\x15topicspace.corpus.nlp\"O\n\x0cTextDocument\x12\x0f\n\x07primary\x18\x01 \x01(\t\x12\x11\n\tsecondary\x18\x02 \x03(\t\x12\x0e\n\x06tokens\x18\x03 \x03(\t\x12\x0b\n\x03url\x18\x04 \x01(\t\"`\n\tTokenStat\x12\r\n\x05token\x18\x01 \x01(\t\x12\x0b\n\x03url\x18\x02 \x01(\t\x12\x11\n\tfrequency\x18\x03 \x01(\x04\x12\x15\n\rdoc_frequency\x18\x04 \x01(\x04\x12\r\n\x05index\x18\x05 \x01(\x04\"w\n\x0eSparseDocument\x12\x0b\n\x03url\x18\x01 \x01(\t\x12\x15\n\rprimary_index\x18\x02 \x01(\x04\x12\x17\n\x0fsecondary_index\x18\x03 \x03(\x04\x12\x13\n\x0btoken_index\x18\x04 \x03(\x04\x12\x13\n\x0btoken_tfidf\x18\x05 \x03(\x02\"D\n\x0f\x43ooccurrenceRow\x12\r\n\x05index\x18\x01 \x01(\x04\x12\x13\n\x0bother_index\x18\x02 \x03(\x04\x12\r\n\x05\x63ount\x18\x03 \x03(\x02\x62\x06proto3')
)




_TEXTDOCUMENT = _descriptor.Descriptor(
  name='TextDocument',
  full_name='topicspace.corpus.nlp.TextDocument',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='primary', full_name='topicspace.corpus.nlp.TextDocument.primary', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='secondary', full_name='topicspace.corpus.nlp.TextDocument.secondary', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='tokens', full_name='topicspace.corpus.nlp.TextDocument.tokens', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='url', full_name='topicspace.corpus.nlp.TextDocument.url', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=36,
  serialized_end=115,
)


_TOKENSTAT = _descriptor.Descriptor(
  name='TokenStat',
  full_name='topicspace.corpus.nlp.TokenStat',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='token', full_name='topicspace.corpus.nlp.TokenStat.token', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='url', full_name='topicspace.corpus.nlp.TokenStat.url', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='frequency', full_name='topicspace.corpus.nlp.TokenStat.frequency', index=2,
      number=3, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='doc_frequency', full_name='topicspace.corpus.nlp.TokenStat.doc_frequency', index=3,
      number=4, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='index', full_name='topicspace.corpus.nlp.TokenStat.index', index=4,
      number=5, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=117,
  serialized_end=213,
)


_SPARSEDOCUMENT = _descriptor.Descriptor(
  name='SparseDocument',
  full_name='topicspace.corpus.nlp.SparseDocument',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='url', full_name='topicspace.corpus.nlp.SparseDocument.url', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='primary_index', full_name='topicspace.corpus.nlp.SparseDocument.primary_index', index=1,
      number=2, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='secondary_index', full_name='topicspace.corpus.nlp.SparseDocument.secondary_index', index=2,
      number=3, type=4, cpp_type=4, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='token_index', full_name='topicspace.corpus.nlp.SparseDocument.token_index', index=3,
      number=4, type=4, cpp_type=4, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='token_tfidf', full_name='topicspace.corpus.nlp.SparseDocument.token_tfidf', index=4,
      number=5, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=215,
  serialized_end=334,
)


_COOCCURRENCEROW = _descriptor.Descriptor(
  name='CooccurrenceRow',
  full_name='topicspace.corpus.nlp.CooccurrenceRow',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='index', full_name='topicspace.corpus.nlp.CooccurrenceRow.index', index=0,
      number=1, type=4, cpp_type=4, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='other_index', full_name='topicspace.corpus.nlp.CooccurrenceRow.other_index', index=1,
      number=2, type=4, cpp_type=4, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='count', full_name='topicspace.corpus.nlp.CooccurrenceRow.count', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=336,
  serialized_end=404,
)

DESCRIPTOR.message_types_by_name['TextDocument'] = _TEXTDOCUMENT
DESCRIPTOR.message_types_by_name['TokenStat'] = _TOKENSTAT
DESCRIPTOR.message_types_by_name['SparseDocument'] = _SPARSEDOCUMENT
DESCRIPTOR.message_types_by_name['CooccurrenceRow'] = _COOCCURRENCEROW
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TextDocument = _reflection.GeneratedProtocolMessageType('TextDocument', (_message.Message,), dict(
  DESCRIPTOR = _TEXTDOCUMENT,
  __module__ = 'nlp_pb2'
  # @@protoc_insertion_point(class_scope:topicspace.corpus.nlp.TextDocument)
  ))
_sym_db.RegisterMessage(TextDocument)

TokenStat = _reflection.GeneratedProtocolMessageType('TokenStat', (_message.Message,), dict(
  DESCRIPTOR = _TOKENSTAT,
  __module__ = 'nlp_pb2'
  # @@protoc_insertion_point(class_scope:topicspace.corpus.nlp.TokenStat)
  ))
_sym_db.RegisterMessage(TokenStat)

SparseDocument = _reflection.GeneratedProtocolMessageType('SparseDocument', (_message.Message,), dict(
  DESCRIPTOR = _SPARSEDOCUMENT,
  __module__ = 'nlp_pb2'
  # @@protoc_insertion_point(class_scope:topicspace.corpus.nlp.SparseDocument)
  ))
_sym_db.RegisterMessage(SparseDocument)

CooccurrenceRow = _reflection.GeneratedProtocolMessageType('CooccurrenceRow', (_message.Message,), dict(
  DESCRIPTOR = _COOCCURRENCEROW,
  __module__ = 'nlp_pb2'
  # @@protoc_insertion_point(class_scope:topicspace.corpus.nlp.CooccurrenceRow)
  ))
_sym_db.RegisterMessage(CooccurrenceRow)


# @@protoc_insertion_point(module_scope)