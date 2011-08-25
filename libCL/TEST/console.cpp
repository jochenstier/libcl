#include <windows.h>

int getKey()
{
    HANDLE lHandle = GetStdHandle(STD_INPUT_HANDLE);

    DWORD lCount;
    GetNumberOfConsoleInputEvents(lHandle, &lCount);
    for (unsigned int i=0; i<lCount; i++)
    {
        INPUT_RECORD IEvent;
        ReadConsoleInput(lHandle, &IEvent, 1, 0);
        switch (IEvent.EventType)
        {
        case KEY_EVENT:
            if (IEvent.Event.KeyEvent.uChar.AsciiChar == 'x')
            {
                return 1;
            }
        }
    }
    return 0;
}

