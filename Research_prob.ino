#include <Adafruit_MLX90614.h>
#include <virtuabotixRTC.h>
#include <Wire.h> 
#include <SPI.h>
#include <SD.h>

Adafruit_MLX90614 mlx = Adafruit_MLX90614();
virtuabotixRTC myRTC(6, 7, 8);

File myFile;
int x;
int val;
 
long pre_time = 0;
//long interval = 30000;
//long interval = 900000;
long interval = 1000; 

void setup() {
  Serial.begin(9600);
  //myRTC.setDS1302Time(30, 05, 11, 1, 1, 3, 2024);
  mlx.begin();
  SD.begin(10);
}

void loop() {
  if(millis() - pre_time >= interval){
     myFile = SD.open("test.txt", FILE_WRITE);
     
     pre_time = millis();
     myRTC.updateTime();

     val = analogRead(A0);
     
     
     Serial.print("Current Date / Time: ");                                                                 
     Serial.print(myRTC.dayofmonth);                                                                   
     Serial.print("/");                                                                                    
     Serial.print(myRTC.month);                                                                           
     Serial.print("/");                                                                                  
     Serial.print(myRTC.year);            
     Serial.print("  ");
     Serial.print(myRTC.hours);                                                                           
     Serial.print(":");                                                                                
     Serial.print(myRTC.minutes);                                                                  
     Serial.print(":");                                                                                   
     Serial.print(myRTC.seconds); 
     Serial.print("  ");
     Serial.print("\tAmbient=");
     Serial.print("  "); 
     Serial.print(mlx.readAmbientTempC());
     Serial.print("  ");
     Serial.print("\tObject=");
     Serial.print("  "); 
     Serial.print(mlx.readObjectTempC());
     Serial.print("  "); 
     Serial.print("  "); 
     Serial.println(val);

     myFile.print("Current Date / Time: "); 
     myFile.print(myRTC.dayofmonth);                                                                 
     myFile.print("/");                                                                                     
     myFile.print(myRTC.month);                                                                             
     myFile.print("/");                                                                                     
     myFile.print(myRTC.year);                                                                             
     myFile.print("  ");
     myFile.print(myRTC.hours);                                                                            
     myFile.print(":");                                                                                    
     myFile.print(myRTC.minutes);                                                                          
     myFile.print(":");                                                                                     
     myFile.print(myRTC.seconds); ;
     myFile.print("  ");
     myFile.print("\tAmbient="); 
     myFile.print("  ");
     myFile.print(mlx.readAmbientTempC());
     myFile.print("  ");
     myFile.print("\tObject=");
     myFile.print("  ");  
     myFile.print(mlx.readObjectTempC());
     myFile.print("  "); 
     myFile.println(val);

     myFile.close();
  }  
}
