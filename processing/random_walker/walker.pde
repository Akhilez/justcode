class Walker {
  
  float x;
  float y;
  
  Walker(){
    x = width/2;
    y = height/2;
  }
  
  void step() {
    float stepX = random(-1,1);
    float stepY = random(-1,1);
    x += stepX;
    y += stepY;;
  }
  
  void display(){
     stroke(0);
     point(x, y);
  }
   
}
