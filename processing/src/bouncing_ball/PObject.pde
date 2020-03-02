abstract class PObject {
  
  private PVector location;
  private PVector velocity;
  private PVector acceleration;
  
  protected float maxVelocity = 10;

  PObject() {
     location = new PVector(width/2, height/2);
     setVelocity(new PVector(0, 0));
     acceleration = new PVector(0, 0);
  }

  protected void setVelocity(PVector velocity) {
    this.velocity = velocity;
    this.velocity.limit(maxVelocity);
  }

  public void update() {
    setVelocity(PVector.add(velocity, acceleration));
    location.add(velocity);
  }

  public abstract void display();

}
