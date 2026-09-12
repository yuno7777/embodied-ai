import unittest
from embodied_ai.engine import Environment, Action

class EngineTests(unittest.TestCase):
 def test_hidden_key_not_visible_from_spawn(self):
    env=Environment(); self.assertTrue(all(e['id']!='exit_key' for c in env.observe()['visible_cells'] for e in c['entities']))

 def test_blocked_move_does_not_change_position(self):
    env=Environment(); before=env.agent.position; env.step(Action('move','west')); self.assertEqual(env.agent.position,before); self.assertEqual(env.invalid_actions,1)

 def test_scripted_success(self):
    env=Environment(); actions=[Action('move','east')]*5+[Action('move','north'),Action('pickup'),Action('move','south'),Action('move','east'),Action('move','east'),Action('open',target_id='exit_door'),Action('move','east')]
    for a in actions: result=env.step(a)
    self.assertTrue(result['done']); self.assertEqual(result['terminal_reason'],'escaped')
