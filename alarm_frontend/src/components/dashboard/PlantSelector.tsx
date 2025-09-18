import { ChevronDown } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Plant } from '@/types/dashboard';

interface PlantSelectorProps {
  plants: Plant[];
  selectedPlant: Plant;
  onPlantChange: (plant: Plant) => void;
  disabled?: boolean;
}

export function PlantSelector({ 
  plants, 
  selectedPlant, 
  onPlantChange, 
  disabled = false 
}: PlantSelectorProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm font-medium text-muted-foreground">Plant:</span>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="outline"
            size="sm"
            disabled={disabled || plants.length <= 1}
            className="gap-2"
          >
            {selectedPlant.name}
            <ChevronDown className="h-4 w-4" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="min-w-[200px]">
          {plants.map((plant) => (
            <DropdownMenuItem
              key={plant.id}
              onClick={() => onPlantChange(plant)}
              className={selectedPlant.id === plant.id ? 'bg-accent' : ''}
            >
              <div className="flex flex-col">
                <span className="font-medium">{plant.name}</span>
                <span className="text-xs text-muted-foreground capitalize">
                  {plant.status}
                </span>
              </div>
            </DropdownMenuItem>
          ))}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
}