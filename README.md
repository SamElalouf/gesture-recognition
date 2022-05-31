# gesture-recognition
A program that uses a neural network to identify hand gestures as inputs commands for spotify and other tools. 
Here is an example of it in action:
<p align="center">
  <img src="demo.gif" alt="animated" />
</p>

The default command logic is set up to issue commands to a Spotify session playing on the browser for the device.
The logic is specified through a function which can be replaced by a custom command logic. Here is the default function:

```ruby
def default_command_logic(class_name):
    if class_name == "C" :
        os.system("spotify play")
    elif class_name == "Palm" :
        os.system("spotify pause")

    else:
        if class_name == "Thumb":
            os.system("spotify volume up 20")
        elif class_name == "Down":
            os.system("spotify volume down 20")
        elif class_name == "Index":
            os.system("spotify next")
        elif class_name == "L":
            os.system("spotify previous")
        elif class_name == "Ok":
            os.system("spotify history")
```

This is a basic framework for how the user can specify their own command logic:

List of command names: ["C", "Palm", "Thumb", "Down", "Index","L", "Ok"]

```ruby
def custom_command_logic(class_name):
    # Do stuff
    pass

```

