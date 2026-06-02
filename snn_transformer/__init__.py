"""
SNN Transformer - PyTorch Implementation

A faithful PyTorch implementation of Eugene Izhikevich's Spiking Neural Network
Transformer from "The Spiking Manifesto" (2025).

This implementation replaces traditional matrix multiplications with Look-Up Table
(LUT) based operations, enabling potential deployment on neuromorphic hardware.
"""

__version__ = "1.0.0"
__author__ = "Based on Eugene Izhikevich's C implementation"
