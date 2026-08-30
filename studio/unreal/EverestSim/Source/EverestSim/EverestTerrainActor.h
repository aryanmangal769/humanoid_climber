#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "EverestTerrainActor.generated.h"

class UProceduralMeshComponent;
class FJsonObject;

UCLASS()
class EVERESTSIM_API AEverestTerrainActor : public AActor
{
    GENERATED_BODY()
public:
    AEverestTerrainActor();
    void ApplyTerrain(const TSharedPtr<FJsonObject>& Terrain);
private:
    UPROPERTY() UProceduralMeshComponent* Mesh;
};
