#pragma once
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "EverestRobotActor.generated.h"
class FJsonObject;
UCLASS()
class EVERESTSIM_API AEverestRobotActor : public AActor
{
    GENERATED_BODY()
public:
    AEverestRobotActor();
    virtual void Tick(float DeltaSeconds) override;
    void ApplyScene(const TSharedPtr<FJsonObject>& Scene);
    void ApplyFrame(const TSharedPtr<FJsonObject>& Frame);
    FVector GetPelvisLocation() const;
private:
    UPROPERTY() USceneComponent* Root;
    UPROPERTY() TMap<FString,USceneComponent*> BodyRoots;
    TMap<FString,FTransform> Targets;
    USceneComponent* EnsureBody(const FString& Name);
};
