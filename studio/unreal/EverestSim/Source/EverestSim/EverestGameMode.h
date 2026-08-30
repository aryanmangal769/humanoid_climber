#pragma once
#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "EverestGameMode.generated.h"
UCLASS() class EVERESTSIM_API AEverestGameMode:public AGameModeBase{GENERATED_BODY()public:AEverestGameMode();virtual void BeginPlay() override;};
