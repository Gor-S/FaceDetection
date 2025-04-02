"""
Compatibility module to handle deprecated functionality in dependencies
"""
import warnings
import importlib

def patch_torchvision_deprecations():
    """
    Patch torchvision deprecation warnings by redirecting imports
    """
    try:
        # Check if we're using a version that has the deprecated module
        import torchvision
        if hasattr(torchvision, 'transforms') and hasattr(torchvision.transforms, 'functional_tensor'):
            # Set up a warning filter to ignore this specific deprecation
            warnings.filterwarnings(
                "ignore", 
                message="The torchvision.transforms.functional_tensor module",
                category=DeprecationWarning
            )
            
            # If v2 is available, we can use that as a replacement
            if hasattr(torchvision.transforms, 'v2'):
                # Monkey patch the old module to use the new one
                import sys
                sys.modules['torchvision.transforms.functional_tensor'] = torchvision.transforms.v2.functional
                print("Patched torchvision.transforms.functional_tensor with v2 implementation")
    except ImportError:
        # If torchvision isn't installed, we don't need to do anything
        pass


def apply_all_patches():
    """Apply all compatibility patches"""
    patch_torchvision_deprecations()
    
    # Suppress common warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")