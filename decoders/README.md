# Decoders

Includes all implemented decoders. They all implement the [decoder](decoder.py) interface. Implemented decoders 
 decode data based on:
 - entropy of data estimated with respect to previous messages.
 - the fact that messages are transmitted using an a priori known protocol.