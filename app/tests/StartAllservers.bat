start cmd.exe /k "%1\bin\skrmedpostctl_start.bat"
ping 1.0.0.1 -n 1 > nul
start cmd.exe /k "%1\bin\wsdserverctl_start.bat"
ping 1.0.0.1 -n 1 > nul
start cmd.exe /k "%1\bin\mmserver14.bat"
ping 1.0.0.1 -n 1 > nul