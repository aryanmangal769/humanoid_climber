#pragma once
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "EverestSnowPatchActor.generated.h"
class UProceduralMeshComponent; class FJsonObject;
UCLASS()
class EVERESTSIM_API AEverestSnowPatchActor : public AActor
{
    GENERATED_BODY()
public:
    AEverestSnowPatchActor();
    void ApplySnowFrame(const TSharedPtr<FJsonObject>& Frame);
private:
    UPROPERTY() UProceduralMeshComponent* Mesh;
};
