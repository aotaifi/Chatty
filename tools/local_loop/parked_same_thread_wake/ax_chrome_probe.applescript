on walkUI(el, depth)
	if depth > 14 then return ""
	set outText to ""
	tell application "System Events"
		try
			set rr to role of el as text
		on error
			set rr to ""
		end try
		if rr is "AXTextArea" or rr is "AXTextField" or rr is "AXWebArea" then
			set dd to ""
			set vv to ""
			try
				set dd to description of el as text
			end try
			try
				set vv to value of el as text
			end try
			set outText to outText & rr & " | DESC=" & dd & " | VALUE=" & vv & linefeed
		end if
		try
			set kids to UI elements of el
		on error
			set kids to {}
		end try
	end tell
	repeat with childEl in kids
		set outText to outText & my walkUI(childEl, depth + 1)
	end repeat
	return outText
end walkUI

tell application "Google Chrome"
	activate
	repeat with w in windows
		set idx to 0
		repeat with t in tabs of w
			set idx to idx + 1
			if URL of t starts with "https://chatgpt.com/c/6ab8b825-2e54-83eb-89b4-707520bbbd36" then
				set active tab index of w to idx
				set index of w to 1
				exit repeat
			end if
		end repeat
	end repeat
end tell

delay 1

tell application "System Events"
	tell process "Google Chrome"
		set targetWindow to front window
	end tell
end tell

return my walkUI(targetWindow, 0)
