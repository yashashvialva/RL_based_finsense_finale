from finsense.models import ActionModel
import json

# Test ActionModel creation like in inference.py
raw = '{"decision": "allow", "approved_amount": 100.0, "reasoning": "test"}'
print('Raw JSON:', repr(raw))

try:
    action_dict = json.loads(raw)
    print('JSON loaded:', action_dict)
    
    action = ActionModel(**action_dict)
    print('ActionModel created successfully:', action)
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()