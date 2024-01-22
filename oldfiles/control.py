import PySimpleGUI as sg
sg.theme('DarkAmber')   # Add a touch of color
layout = [
    [sg.Text("Comuter Vision Control Station")],
    [sg.Button("Edge")],
    [sg.Button("Motion_Detection")],
    [sg.Button("Exit")]

]
controlpanel = sg.Window("CV Control", layout)

while True:
    event, values = controlpanel.read()
    if event == sg.WIN_CLOSED or event == 'Exit': # if user closes window or clicks cancel
        break
    elif event == 'Edge':
        pass
    elif event == 'Motion_Detection':
        pass

controlpanel.close()


