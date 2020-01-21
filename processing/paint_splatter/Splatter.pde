import java.util.Random;

class Splatter {
  
  int x;
  int y;
  
  int sd;
  int mean;
  
  Random generator;
  
  Splatter () {
    x = width / 2;
    y = height / 2;
    sd = 5;
    generator = new Random();
  }
  
  void paint() {
    
    if (mousePressed) {
    
      x = mouseX;
      y = mouseY;
      
      float positionX =(float) generator.nextGaussian();
      positionX = positionX * sd + x;
      
      float positionY =(float) generator.nextGaussian();
      positionY = positionY * sd + y;
      
      noStroke();
      fill(0, 50);
      ellipse(positionX, positionY, 5, 5);
      
    }
    
  }
    
    
}
