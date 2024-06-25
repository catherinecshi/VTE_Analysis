class trajectory:
    _instances = []
    _next_id = 1
    
    def __init__(self, x_vals, y_vals, trajectory_ID):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.trajectory_ID += 1
        
        trajectory._instances.append(self)
    
    @classmethod
    def get_instace_by_id(cls, instance_id):
        return cls._instances_by_id.get(instance_id)
    
    @classmethod
    def find_by_property(cls, property, value):
        return [instance for instance in cls.instances if getattr(instance, property, None) == value]