import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to sys.path so we can import modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from agent.engine import AgentSession

def test_agent_concurrency():
    print("Loading configuration...")
    try:
        cfg = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # We will mock subprocess.Popen
    mock_proc = MagicMock()
    mock_stdout = MagicMock()
    # readline returns READY on first call, and a JSON response on subsequent calls
    mock_stdout.readline.side_effect = [
        "READY\n",
        '{"text": "Hello world!"}\n',
        '{"text": "Hello again!"}\n'
    ]
    mock_proc.stdout = mock_stdout
    mock_proc.stdin = MagicMock()

    # Reset static class attributes to ensure fresh test state
    AgentSession._active_session = None
    AgentSession._ref_count = 0

    print("Entering first AgentSession...")
    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        with AgentSession(cfg) as session1:
            print("First session entered.")
            
            # Verify Popen was called
            mock_popen.assert_called_once()
            assert AgentSession._active_session is session1
            assert AgentSession._ref_count == 1
            assert session1._proc is mock_proc

            print("Entering second (nested) AgentSession...")
            # Popen should NOT be called again
            mock_popen.reset_mock()
            with AgentSession(cfg) as session2:
                print("Second session entered.")
                mock_popen.assert_not_called()
                
                # Check that session2 reused session1's process
                assert AgentSession._active_session is session1
                assert AgentSession._ref_count == 2
                assert session2._proc is mock_proc

                # Test generation works in nested session
                print("Testing generation in second session...")
                res = session2.generate([{"role": "user", "content": "hi"}])
                assert res == "Hello world!"
                
            # After exiting session2, ref_count should decrement but subprocess should NOT be killed
            print("Exited second session.")
            assert AgentSession._active_session is session1
            assert AgentSession._ref_count == 1
            assert session1._proc is mock_proc
            assert session2._proc is None
            mock_proc.stdin.close.assert_not_called()
            mock_proc.wait.assert_not_called()
            mock_proc.kill.assert_not_called()

            # Test generation still works in session1
            print("Testing generation in first session...")
            res = session1.generate([{"role": "user", "content": "hi again"}])
            assert res == "Hello again!"

        # After exiting session1, ref_count should reach 0 and cleanup should happen
        print("Exited first session.")
        assert AgentSession._active_session is None
        assert AgentSession._ref_count == 0
        assert session1._proc is None
        
        # Verify cleanup calls were made
        mock_proc.stdin.close.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=30)
        
    print("Testing mismatched config name error...")
    # Setup active session again
    AgentSession._active_session = session1
    AgentSession._ref_count = 1
    session1._proc = mock_proc
    
    # Try starting a different backend name
    import copy
    cfg_different = copy.deepcopy(cfg)
    cfg_different.agent_backend_name = "different_agent_backend"
    
    try:
        with AgentSession(cfg_different):
            raise AssertionError("Should not allow starting a different agent backend concurrently!")
    except RuntimeError as e:
        print(f"Caught expected RuntimeError: {e}")
        assert "Cannot start agent backend" in str(e)

    # Clean up static state
    AgentSession._active_session = None
    AgentSession._ref_count = 0

    print("\nAll concurrency tests PASSED!")

if __name__ == "__main__":
    test_agent_concurrency()
