uint8 DAMP=0     # motors damped ("power off"; start state; fall)
uint8 STAND=1    # policy balancing in place, zero velocity, default arms  (= "locomotion paused")
uint8 LOCO=2     # keyboard / UI locomotion teleop (lease holder)
uint8 MANIP=3    # camera upper-body + hand teleop; legs hold stance, cmd_vel forced 0
uint8 NAV=4      # Nav2 drives cmd_vel
uint8 AUTO=5     # mission/skills drive: Nav2 cmd_vel + /g1/auto/* arm & hand targets
uint8 HOLD=6     # "Stop": zero velocity, arms frozen where they are, hands keep their command, task paused,
                 # still balancing. Left only by an explicit resume (-> STAND, or -> AUTO if held from AUTO).
uint8 mode
string source    # who requested it: keyboard, ui:<device_id>, mission, safety, watchdog, estop_hw, lease, nav
string detail
