#pragma once
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "EverestSnowfallActor.generated.h"
class UInstancedStaticMeshComponent;
UCLASS()
class EVERESTSIM_API AEverestSnowfallActor : public AActor
{
    GENERATED_BODY()
public:
    AEverestSnowfallActor();
    virtual void BeginPlay() override;
    virtual void Tick(float DeltaSeconds) override;
    void SetWeather(float SnowfallMmH,float WindMps,float WindDirectionDeg,float Visibility);
    void SetFocus(FVector P){Focus=P;}
private:
    UPROPERTY() UInstancedStaticMeshComponent* Flakes;
    TArray<FVector> Positions; FVector Focus=FVector::ZeroVector; float FallSpeed=170.f,WindCmS=0.f,WindDeg=0.f,Intensity=0.4f,Visibility=1.f;
    void ResetFlake(int32 I,bool RandomZ);
};
