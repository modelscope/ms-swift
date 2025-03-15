from typing import Tuple,Any, Optional
class TOOL_CALL:
    def __call__(self, completion: str) -> Tuple[Any, bool, Optional[float]]:
        raise NotImplementedError


tools = {
    
}