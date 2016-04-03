# !/usr/bin/python

# diecontrol.py
# version 1.0, 2/23/16
# mouser@donationcoder.com

"""
Functions for controlling hardware (die roller)
"""


import time



class DieRollerHardware:
    """Class to supervise die roller hardware."""

    def __init__(self):
        """Initialize any hardware?"""
        # options
        self.spinTimeMs = 250
        self.extraSettleWaitTimeMs = 500
        self.maxSettleWaitTimeMs = 2000
        self.waitForSettle = False

    def set_spinTimeMs(self, val):
        self.spinTimeMs = val
    def set_waitForSettle(self, val):
        self.waitForSettle = val


    def connect(self):
        """Connection to hardware."""
        pass


    def disconnect(self):
        """Disconnect from hardware."""
        pass


    def spin(self):
        """Run spinner."""
        # run it
        retv = self.hardwareSpin()
        if (not retv):
            print "WARNING: Trying to reinitialize device after failed spin command."
            time.sleep(1)
            self.disconnect()
            time.sleep(1)
            self.connect()
            retv = self.hardwareSpin()
            if (not retv):
                print "ERROR: Spin command failed."
        # wait?
        if (self.waitForSettle):
            # return after it's probably settled
            #print "Waiting after settle."
            waitForMs = max(self.spinTimeMs + self.extraSettleWaitTimeMs, self.maxSettleWaitTimeMs)
            time.sleep(float(waitForMs)/1000.0)
            #print "done waiting."


    def hardwareSpin(self):
        """Derived class should handle this."""
        raise "ERROR: Called hardwareSpin from base DieRollerHardware class."
        #return False





class DieRollerHardware_ArduinoRelay(DieRollerHardware):
    """Derived class for using arduino relay to turn on spinner."""

    def __init__(self):
        DieRollerHardware.__init__(self)
        # default connection parameters (these should be overridden by caller)
        self.comPort = 'COM6'
        # this param should stay unchanged
        self.baudRate = 9600
        #self.baudRate = 2400
        self.arduinoStepMs = 250
        self.waitAfterConnect = 2000
        # timeouts -- could be 0 for unlimited, or try something else
        self.serialReadTimeout = 2
        self.serialWriteTimeout = 2
        # init
        self.serialPipe = None

    def set_comPort(self, val):
        self.comPort = val


    def connect(self):
        """Derived implementation."""
        import serial
        print "Connecting to Arduino using com port %s." % self.comPort
        self.serialPipe = serial.Serial(self.comPort, self.baudRate, timeout = self.serialReadTimeout, writeTimeout=self.serialWriteTimeout)
        #self.serialPipe = serial.Serial(self.comPort, self.baudRate)
        # we need to wait a while after connect resets arduino
        time.sleep(float(self.waitAfterConnect)/1000.0)

    def disconnect(self):
        """Derived implementation - nothing needed."""
        self.serialPipe.close()


    def hardwareSpin(self):
        """Derived implementation."""
        # the arduino code running expects a single character ranging from '1' to '9' telling it how many arduinoStepMs (250ms) time periods to run
        steps = self.spinTimeMs / self.arduinoStepMs
        # test
        #steps = 1
        #
        charCode = 48 + steps
        charCodeChr = chr(48 + steps)
        #print "Sending sleep serial code: %s" % charCodeChr
        retv = self.serialPipe.write(charCodeChr)
        #print "Arduino serial write returned: %d" % retv

        # see http://stackoverflow.com/questions/20360432/arduino-serial-timeouts-after-several-serial-writes
        if (True):
            # Flush input buffer, if there is still some unprocessed data left
            # Otherwise the APM 2.5 control boards stucks after some command
            self.serialPipe.flush()       # Try to send old message
            self.serialPipe.flushInput()  # Delete what is still inside the buffer

        retline = self.serialPipe.readline()
        #print "Arduino code reply: %s" % retline
        if (len(retline)<2):
            print "Bad arduino code reply from hardware spin command (len %d)." % len(retline)
            return False

        return True



"""
// Arduino Code used to drive relay for certain periods of time
// Used in dicer project.
// mouser@donationcoder.com - 2/25/16


// select the pin for the relay
int relayPin = 8;
int baudrate = 9600;
// how long to run increment arg passed to us
int perStepTimeMs = 250;
// some randomness
int randomRangeTimeMs = perStepTimeMs * 2;






void setup() {
  // initialize what pin we use
  pinMode(relayPin,OUTPUT);    // declare the LED's pin as output
  Serial.begin(baudrate);        // connect to the serial port

  // init random number generator -- see https://www.arduino.cc/en/Reference/Random
  randomSeed(analogRead(0));

  // Wait until Serial is ready?
  while (! Serial);
}






void loop () {
  int val;

  if (Serial.available()) {

    // read char from serial port
    val = Serial.read();

    // If the stored value is a single-digit number, that char - '0' is how many time steps to run for
    // if we get something else, report an error
    if (val > '0' && val <= '9' ) {
      int charcode = val-'0';
      int runTimeMs = charcode * perStepTimeMs;

      // we should add a little randomness to runtime don't you think?
      runTimeMs += random(0, randomRangeTimeMs);

      // let caller know we got the command ok
      Serial.println("OK. Engaging relay.");

      // engage relay
      digitalWrite(relayPin,HIGH);

      // wait
      delay(runTimeMs);

      // turn off relay
      digitalWrite(relayPin, LOW);
    }
    else {
      // error, unknown command received
      Serial.print("ERROR. Unknown character command received: ");
      Serial.println(val);
    }
  }
}
"""