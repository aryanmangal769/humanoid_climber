#pragma once
#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "EverestOperatorPawn.generated.h"
class USpringArmComponent; class UCameraComponent; class AEverestWorldManager;
UCLASS()
class EVERESTSIM_API AEverestOperatorPawn : public APawn
{
    GENERATED_BODY()
public:
    AEverestOperatorPawn();
    virtual void BeginPlay() override; virtual void Tick(float DeltaSeconds) override; virtual void SetupPlayerInputComponent(UInputComponent* PlayerInputComponent) override;
private:
    UPROPERTY() USceneComponent* Root; UPROPERTY() USpringArmComponent* Arm; UPROPERTY() UCameraComponent* Camera; UPROPERTY() AEverestWorldManager* Manager=nullptr;
    float Forward=0,Lateral=0,YawRate=0,OrbitYaw=25,OrbitPitch=-18; bool bPaused=true; float SendAccumulator=0;
    void SetForward(float V){Forward=V;} void SetLateral(float V){Lateral=V;} void SetYaw(float V){YawRate=V;} void CameraYaw(float V); void CameraPitch(float V); void CameraZoom(float V); void TogglePause();
};
