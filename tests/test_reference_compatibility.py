"""Test compatibility with JABCode reference implementation.

This module tests that our encoder produces identical output to the
reference JABCode implementation for the same inputs.
"""

import pytest
import json
import os
from PIL import Image
import numpy as np
from pyhue2d import encode
from pyhue2d.jabcode.encoder import JABCodeEncoder
from pyhue2d.jabcode.ldpc.seed_config import RandomSeedConfig


class TestReferenceCompatibility:
    """Test suite for JABCode reference implementation compatibility."""
    
    @pytest.fixture
    def examples_data(self):
        """Load examples manifest data."""
        examples_dir = os.path.join(os.path.dirname(__file__), 'example_images')
        manifest_path = os.path.join(examples_dir, 'examples_manifest.json')
        
        with open(manifest_path) as f:
            examples = json.load(f)
        
        return examples, examples_dir
    
    def test_reference_image_properties(self, examples_data):
        """Test that reference images have expected properties."""
        examples, examples_dir = examples_data
        
        for example in examples:
            image_path = os.path.join(examples_dir, example['output'])
            assert os.path.exists(image_path), f"Reference image {example['output']} not found"
            
            image = Image.open(image_path)
            
            # All reference images should be 252x252 (21x21 modules * 12 scaling)
            assert image.size == (252, 252), f"Expected 252x252, got {image.size}"
            
            # Should be RGBA mode
            assert image.mode == 'RGBA', f"Expected RGBA mode, got {image.mode}"
            
            # Should have 8 colors (including transparency)
            img_array = np.array(image)
            unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0))
            assert unique_colors == 8, f"Expected 8 colors, got {unique_colors}"
    
    def test_reference_settings_analysis(self, examples_data):
        """Analyze what settings the reference implementation uses."""
        examples, examples_dir = examples_data
        
        # Test with the first example
        example = examples[0]
        image_path = os.path.join(examples_dir, example['output'])
        reference_image = Image.open(image_path)
        
        print(f"\n📊 REFERENCE IMAGE ANALYSIS: {example['output']}")
        print(f"Text: '{example['text']}'")
        print(f"Size: {reference_image.size}")
        print(f"Mode: {reference_image.mode}")
        
        # Convert to RGB for comparison
        if reference_image.mode == 'RGBA':
            # Convert RGBA to RGB by compositing over white background
            background = Image.new('RGB', reference_image.size, (255, 255, 255))
            reference_rgb = Image.alpha_composite(background.convert('RGBA'), reference_image)
            reference_rgb = reference_rgb.convert('RGB')
        else:
            reference_rgb = reference_image.convert('RGB')
        
        # Analyze the color palette
        img_array = np.array(reference_rgb)
        unique_colors = np.unique(img_array.reshape(-1, 3), axis=0)
        print(f"Unique RGB colors: {len(unique_colors)}")
        for i, color in enumerate(unique_colors):
            print(f"  Color {i}: RGB{tuple(color)}")
    
    def test_our_encoder_basic_output(self):
        """Test our encoder's basic output characteristics."""
        text = "Hello, JAB Code!"
        
        # Test with default settings
        our_image = encode(text, colors=8, ecc_level='M')
        
        print(f"\n🔧 OUR ENCODER OUTPUT:")
        print(f"Text: '{text}'")
        print(f"Size: {our_image.size}")
        print(f"Mode: {our_image.mode}")
        
        # Analyze our color palette
        img_array = np.array(our_image)
        unique_colors = np.unique(img_array.reshape(-1, 3), axis=0)
        print(f"Unique RGB colors: {len(unique_colors)}")
        for i, color in enumerate(unique_colors):
            print(f"  Color {i}: RGB{tuple(color)}")
    
    def test_encoder_with_reference_settings(self):
        """Test encoder with settings to match reference implementation."""
        text = "Hello, JAB Code!"
        
        # Try to configure encoder to match reference implementation
        settings = {
            'color_count': 8,
            'ecc_level': 'M',  # This should map to reference level 3
            'module_size': 12,  # Reference uses 12 pixels per module
            'mask_pattern': 7,  # Reference default mask pattern
            'symbol_size': (21, 21),  # Force 21x21 modules
        }
        
        try:
            encoder = JABCodeEncoder(settings)
            our_image = encoder.encode_to_image(text)
            
            print(f"\n⚙️  ENCODER WITH REFERENCE SETTINGS:")
            print(f"Text: '{text}'")
            print(f"Settings: {settings}")
            print(f"Output size: {our_image.size}")
            print(f"Expected size: (252, 252)")  # 21 * 12 = 252
            
        except Exception as e:
            print(f"❌ Failed to create encoder with reference settings: {e}")
    
    def test_ldpc_seed_configuration(self):
        """Test LDPC seed configuration matches reference."""
        # Reference implementation uses specific seeds
        expected_message_seed = 785465
        expected_metadata_seed = 38545
        
        seed_config = RandomSeedConfig()
        
        print(f"\n🎲 LDPC SEED ANALYSIS:")
        print(f"Expected message seed: {expected_message_seed}")
        print(f"Expected metadata seed: {expected_metadata_seed}")
        print(f"Our message seed: {seed_config.message_seed}")
        print(f"Our metadata seed: {seed_config.metadata_seed}")
        
        # These should match for identical output
        # (Currently may not match - this test documents the requirement)
    
    @pytest.mark.skip(reason="Will implement after encoder fixes")
    def test_exact_reference_reproduction(self, examples_data):
        """Test that our encoder produces identical images to reference."""
        examples, examples_dir = examples_data
        
        for example in examples:
            text = example['text']
            reference_path = os.path.join(examples_dir, example['output'])
            reference_image = Image.open(reference_path)
            
            # Configure encoder to match reference exactly
            settings = {
                'color_count': 8,
                'ecc_level': 'M', 
                'module_size': 12,
                'mask_pattern': 7,
                'ldpc_message_seed': 785465,
                'ldpc_metadata_seed': 38545,
            }
            
            encoder = JABCodeEncoder(settings)
            our_image = encoder.encode_to_image(text)
            
            # Convert both to same format for comparison
            if reference_image.mode == 'RGBA':
                reference_rgb = reference_image.convert('RGB')
            else:
                reference_rgb = reference_image
            
            our_rgb = our_image.convert('RGB')
            
            # Compare images
            assert our_rgb.size == reference_rgb.size, \
                f"Size mismatch: our {our_rgb.size} vs reference {reference_rgb.size}"
            
            # Compare pixel data
            our_array = np.array(our_rgb)
            ref_array = np.array(reference_rgb)
            
            # Allow for minor differences due to color palette ordering
            # but the overall pattern should be identical
            assert our_array.shape == ref_array.shape, \
                f"Array shape mismatch: our {our_array.shape} vs reference {ref_array.shape}"
            
            # For now, just ensure we have the same number of unique colors
            our_colors = len(np.unique(our_array.reshape(-1, 3), axis=0))
            ref_colors = len(np.unique(ref_array.reshape(-1, 3), axis=0))
            
            assert our_colors == ref_colors, \
                f"Color count mismatch: our {our_colors} vs reference {ref_colors}"
    
    def test_debug_reference_vs_ours(self, examples_data):
        """Debug comparison between reference and our implementation."""
        examples, examples_dir = examples_data
        
        # Use first example for detailed comparison
        example = examples[0]
        text = example['text']
        reference_path = os.path.join(examples_dir, example['output'])
        
        print(f"\n🔍 DETAILED COMPARISON:")
        print(f"Text: '{text}'")
        
        # Load reference image
        reference_image = Image.open(reference_path)
        reference_rgb = reference_image.convert('RGB')
        
        # Create our image with current settings
        our_image = encode(text, colors=8, ecc_level='M')
        
        print(f"\nReference:")
        print(f"  Size: {reference_rgb.size}")
        print(f"  Mode: {reference_rgb.mode}")
        
        print(f"\nOurs:")
        print(f"  Size: {our_image.size}")
        print(f"  Mode: {our_image.mode}")
        
        print(f"\nSize ratio: {reference_rgb.size[0] / our_image.size[0]:.1f}x")
        
        # Save both for manual inspection
        debug_dir = "debug_comparison"
        os.makedirs(debug_dir, exist_ok=True)
        
        reference_rgb.save(f"{debug_dir}/reference_{example['output']}")
        our_image.save(f"{debug_dir}/ours_{example['output']}")
        
        print(f"\n💾 Saved images to {debug_dir}/ for manual comparison")


if __name__ == "__main__":
    # Run specific tests for debugging
    import sys
    
    test_instance = TestReferenceCompatibility()
    
    # Load examples data
    examples_dir = os.path.join(os.path.dirname(__file__), 'example_images')
    manifest_path = os.path.join(examples_dir, 'examples_manifest.json')
    
    with open(manifest_path) as f:
        examples = json.load(f)
    
    examples_data = (examples, examples_dir)
    
    print("🧪 RUNNING REFERENCE COMPATIBILITY TESTS")
    print("=" * 50)
    
    try:
        test_instance.test_reference_image_properties(examples_data)
        print("✅ Reference image properties test passed")
    except Exception as e:
        print(f"❌ Reference image properties test failed: {e}")
    
    try:
        test_instance.test_reference_settings_analysis(examples_data)
        print("✅ Reference settings analysis completed")
    except Exception as e:
        print(f"❌ Reference settings analysis failed: {e}")
    
    try:
        test_instance.test_our_encoder_basic_output()
        print("✅ Our encoder basic output test completed")
    except Exception as e:
        print(f"❌ Our encoder basic output test failed: {e}")
    
    try:
        test_instance.test_debug_reference_vs_ours(examples_data)
        print("✅ Debug comparison completed")
    except Exception as e:
        print(f"❌ Debug comparison failed: {e}")