"""Progress bar utilities for displaying processing status in terminal"""
import sys
import time
from typing import Optional, Union, Iterable, Iterator
from . import logger

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

get_logger = logger.get_logger(module_name="progress_bar")

class ProgressBar:
    """Class for tracking and displaying progress in terminal"""
    
    def __init__(self, total: int, desc: str = "", unit: str = "it", 
                 color: str = "green", disable: bool = False):

        self.total = total
        self.desc = desc
        self.unit = unit
        self.color = color
        self.disable = disable
        self.logger = get_logger
        
        # Create tqdm progress bar if available
        if _TQDM_AVAILABLE and not self.disable:
            self.pbar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                colour=color,
                file=sys.stdout,
                leave=True
            )
        else:
            self.pbar = None
            if not self.disable:
                self.logger.warning("tqdm not available. Install with 'pip install tqdm' for progress bars.")
                print(f"{desc} [started] - {total} {unit}")
                self.start_time = time.time()
        
        self.current = 0
        
    def update(self, n: int = 1) -> None:

        self.current += n
        if self.pbar:
            self.pbar.update(n)
        elif not self.disable and self.current % max(1, int(self.total * 0.05)) == 0:
            # Without tqdm, print an update every 5% of progress
            elapsed = time.time() - self.start_time
            percentage = min(100, int(100 * self.current / self.total))
            if self.total > 0:
                print(f"\r{self.desc} [{percentage}%] - {self.current}/{self.total} {self.unit} "
                      f"(elapsed: {elapsed:.1f}s)", end="")
    
    def close(self) -> None:
        """Close and finalize the progress bar"""
        if self.pbar:
            self.pbar.close()
        elif not self.disable:
            elapsed = time.time() - self.start_time
            print(f"\r{self.desc} [completed] - {self.current}/{self.total} {self.unit} "
                  f"(total time: {elapsed:.1f}s)")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def progress_iterate(iterable: Iterable, desc: str = "", total: Optional[int] = None, 
                    unit: str = "it", color: str = "green", disable: bool = False) -> Iterator:

    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            total = 0  # Unknown total
    
    with ProgressBar(total=total, desc=desc, unit=unit, color=color, disable=disable) as pbar:
        for item in iterable:
            yield item
            pbar.update()
