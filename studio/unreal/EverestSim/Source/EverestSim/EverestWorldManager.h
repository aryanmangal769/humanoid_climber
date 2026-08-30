#pragma once
#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "IWebSocket.h"
#include "EverestWorldManager.generated.h"
class AEverestTerrainActor; class AEverestSnowPatchActor; class AEverestRobotActor; class AEverestSnowfallActor;
UCLASS()
class EVERESTSIM_API AEverestWorldManager : public AActor
{
    GENERATED_BODY()
public:
    AEverestWorldManager();
    virtual void BeginPlay() override;
    void SendRobotCommand(float Forward,float Lateral,float YawRate);
    void SetSimulationPaused(bool bPaused);
    AEverestRobotActor* GetRobot() const { return Robot; }
private:
    UPROPERTY() AEverestTerrainActor* Terrain=nullptr;
    UPROPERTY() AEverestTerrainActor* MacroTerrain=nullptr;
    UPROPERTY() AEverestSnowPatchActor* Snow=nullptr;
    UPROPERTY() AEverestRobotActor* Robot=nullptr;
    UPROPERTY() AEverestSnowfallActor* Snowfall=nullptr;
    TSharedPtr<IWebSocket> Socket;
    void OnMessage(const FString& Message);
    void SendControl(const FString& Action,const FString& JsonValue);
};
